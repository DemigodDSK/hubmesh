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
import hmac
import json
import os
from dataclasses import dataclass

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


_read_only = False   # set by main(); write tools refuse when True


@mcp.tool()
def index_corpus(name: str, documents: list[dict]) -> dict:
    """Index documents into a named corpus: embeds them, builds the
    entity knowledge graph (spaCy NER), and persists everything to
    disk. `documents` is a list of {"id": str, "text": str}. Re-using
    an existing name replaces that corpus. Indexing cost is paid once —
    queries afterwards are ~milliseconds."""
    if _read_only:
        return {"error": "this server is read-only: indexing is disabled "
                         "(started with --read-only, or serving through a "
                         "tunnel without --allow-writes)"}
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


# ---- network security ------------------------------------------------

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


TUNNEL_AUTH_NOTICE = (
    "tunnel mode: every caller must present the bearer token. If your "
    "connector cannot send headers and you inject the token at the tunnel "
    "edge, the edge MUST authenticate callers first — injecting it for "
    "anonymous traffic makes every corpus publicly readable (read-only "
    "prevents replacement, not disclosure; get_document returns full text)."
)


@dataclass
class SecurityPolicy:
    api_key: str | None
    read_only: bool
    notice: str | None = None    # operator-facing caveat, printed at startup


def resolve_security(transport: str, host: str, allow_tunnel: bool,
                     api_key: str | None, read_only: bool,
                     allow_writes: bool) -> SecurityPolicy:
    """Decide the serving security posture, refusing insecure setups.

    Rules (external review, batch 2): any network exposure beyond
    loopback — a non-loopback bind or a tunnel — REQUIRES an API key;
    tunnel mode additionally defaults to read-only unless --allow-writes
    is explicit. stdio has a local process trust boundary and takes no
    key. A key supplied on loopback is still enforced (belt on localhost
    is allowed, just not demanded)."""
    if transport == "stdio":
        return SecurityPolicy(api_key=None, read_only=read_only)
    exposed = allow_tunnel or host not in _LOOPBACK_HOSTS
    if exposed and not api_key:
        raise SystemExit(
            "refusing to serve: this configuration exposes read/write "
            "corpus tools beyond localhost without authentication. Set "
            "--api-key or HUBMESH_API_KEY (any strong secret), or bind "
            "to 127.0.0.1 without --allow-tunnel.")
    ro = read_only or (allow_tunnel and not allow_writes)
    return SecurityPolicy(api_key=api_key, read_only=ro,
                          notice=TUNNEL_AUTH_NOTICE if allow_tunnel else None)


class BearerAuthASGI:
    """Minimal ASGI middleware: every HTTP request must carry
    `Authorization: Bearer <key>` (constant-time comparison). Non-HTTP
    scopes (lifespan) pass through untouched."""

    def __init__(self, app, api_key: str):
        self.app = app
        self._expect = f"Bearer {api_key}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        auth = b""
        for k, v in scope.get("headers", []):
            if k.lower() == b"authorization":
                auth = v
                break
        if not hmac.compare_digest(auth, self._expect):
            body = json.dumps({"error": "unauthorized: missing or invalid "
                                        "Authorization bearer token"}).encode()
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"www-authenticate", b"Bearer")]})
            await send({"type": "http.response.body", "body": body})
            return
        return await self.app(scope, receive, send)


def build_sse_app(api_key: str | None):
    """The SSE ASGI app, auth-wrapped when a key is configured —
    factored out of main() so the transport can be tested for real."""
    app = mcp.sse_app()
    return BearerAuthASGI(app, api_key) if api_key else app


def main():
    import argparse
    import threading
    ap = argparse.ArgumentParser(
        prog="hubmesh-mcp",
        description="hubmesh MCP operator server. stdio by default; "
                    "--transport sse serves HTTP+SSE natively (no "
                    "gateway process needed). Serving beyond localhost "
                    "requires --api-key / HUBMESH_API_KEY.")
    ap.add_argument("--transport", choices=["stdio", "sse"],
                    default="stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--allow-tunnel", action="store_true",
                    help="accept forwarded Host headers (disables "
                         "DNS-rebinding protection) — required behind "
                         "ngrok-style tunnels, which otherwise get 421 "
                         "Misdirected Request. Requires an API key and "
                         "implies --read-only unless --allow-writes.")
    ap.add_argument("--api-key", default=os.environ.get("HUBMESH_API_KEY"),
                    help="bearer token clients must send (env: "
                         "HUBMESH_API_KEY). Mandatory for non-loopback "
                         "binds and tunnels; optional but enforced on "
                         "loopback.")
    ap.add_argument("--read-only", action="store_true",
                    help="disable indexing/replacement tools")
    ap.add_argument("--allow-writes", action="store_true",
                    help="keep write tools enabled behind a tunnel "
                         "(default there is read-only)")
    args = ap.parse_args()

    policy = resolve_security(args.transport, args.host, args.allow_tunnel,
                              args.api_key, args.read_only,
                              args.allow_writes)
    global _read_only
    _read_only = policy.read_only
    if policy.notice:
        import sys
        print(f"hubmesh-mcp: {policy.notice}", file=sys.stderr)

    # Warm up off the serving thread: the first tool call must not pay
    # the ~5-10s model cold start — connector clients (e.g. Perplexity)
    # drop SSE tool calls in exactly that window.
    threading.Thread(target=lambda: _mgr().warmup(), daemon=True).start()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return

    mcp.settings.host = args.host
    mcp.settings.port = args.port
    if args.allow_tunnel:
        # Rebinding protection rejects forwarded Host headers with 421;
        # behind a tunnel the bearer auth (mandatory here) is the defense.
        from mcp.server.transport_security import TransportSecuritySettings
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=False)

    import uvicorn
    uvicorn.run(build_sse_app(policy.api_key),
                host=args.host, port=args.port)


if __name__ == "__main__":
    main()
