"""Named-corpus persistence — the substrate for hubmesh-mcp.

Serialises a corpus (documents + vectors + EntityKG) to plain JSON/NPZ.
Deliberately no binary object serialisation: corpus directories are
safe to inspect, diff, and share, and loading one can never execute
code. Layout under `<root>/<name>/`:

    meta.json      — format version, counts
    docs.jsonl     — one {id, text, metadata} per line
    vectors.npz    — doc_ids array + float32 embedding matrix
    kg.json        — nodes/edges + the EntityKG aux maps

A loaded corpus rebuilds its Planner (and the cached PPR transition
matrix) on first use — that cost is paid once per process, not per
query, same as constructing a Planner by hand.
"""
from __future__ import annotations
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import numpy as np
import networkx as nx

from .types import Document
from .adapters.inmemory import InMemoryStore
from .kg import EntityKG
from .planner import Planner, PlannerConfig

FORMAT_VERSION = 1
DEFAULT_ROOT = Path.home() / ".hubmesh" / "corpora"


# ---------------------------------------------------------------------
# EntityKG ↔ plain-dict serialisation (version-stable, no nx helpers —
# node_link_data's signature has churned across networkx releases)
# ---------------------------------------------------------------------

def kg_to_dict(kg: EntityKG) -> dict:
    return {
        "nodes": [[n, d] for n, d in kg.graph.nodes(data=True)],
        "edges": [[u, v, d] for u, v, d in kg.graph.edges(data=True)],
        "doc_to_entities": {k: sorted(v) for k, v in kg.doc_to_entities.items()},
        "entity_to_docs": {k: sorted(v) for k, v in kg.entity_to_docs.items()},
        "entity_canonical_to_node": kg.entity_canonical_to_node,
        "entity_node_to_label": kg.entity_node_to_label,
        "alias_to_node": kg.alias_to_node,
    }


def kg_from_dict(d: dict) -> EntityKG:
    G = nx.Graph()
    for n, attrs in d["nodes"]:
        G.add_node(n, **attrs)
    for u, v, attrs in d["edges"]:
        G.add_edge(u, v, **attrs)
    return EntityKG(
        graph=G,
        doc_to_entities={k: set(v) for k, v in d["doc_to_entities"].items()},
        entity_to_docs={k: set(v) for k, v in d["entity_to_docs"].items()},
        entity_canonical_to_node=d["entity_canonical_to_node"],
        entity_node_to_label=d["entity_node_to_label"],
        alias_to_node=d.get("alias_to_node", {}),
    )


@dataclass
class CorpusManager:
    """Create, persist, and reopen named corpora.

    `embed` maps one text → np.ndarray. Defaults to a lazily-loaded
    sentence-transformers model (HUBMESH_EMBED_MODEL env var overrides
    the model name). Inject your own callable for custom embeddings or
    for tests. `nlp` is the spaCy pipeline used when a KG has to be
    built and none is supplied.
    """
    root: Path = field(default_factory=lambda: DEFAULT_ROOT)
    embed: Callable[[str], np.ndarray] | None = None
    nlp: object | None = None
    _planners: dict = field(default_factory=dict, repr=False)
    _embed_lock: threading.Lock = field(default_factory=threading.Lock,
                                        repr=False)

    def __post_init__(self):
        self.root = Path(self.root).expanduser()

    # ---- embedding -------------------------------------------------

    def _get_embed(self) -> Callable[[str], np.ndarray]:
        if self.embed is None:
            with self._embed_lock:
                if self.embed is None:
                    import os
                    from .entity_linker import make_st_embedder
                    model = os.environ.get("HUBMESH_EMBED_MODEL",
                                           "all-MiniLM-L6-v2")
                    # Load from the local HF cache without the hub
                    # freshness-check round-trips that dominate cold
                    # start on slow networks; fall back online for the
                    # first-ever model download.
                    set_offline = "HF_HUB_OFFLINE" not in os.environ
                    if set_offline:
                        os.environ["HF_HUB_OFFLINE"] = "1"
                    try:
                        batched = make_st_embedder(model)
                    except Exception:
                        if not set_offline:
                            raise
                        os.environ.pop("HF_HUB_OFFLINE", None)
                        set_offline = False
                        batched = make_st_embedder(model)
                    finally:
                        if set_offline:
                            os.environ.pop("HF_HUB_OFFLINE", None)
                    self.embed = lambda t: batched([t])[0]
        return self.embed

    def warmup(self, corpora: bool = True) -> dict:
        """Front-load the expensive lazy imports — embedding model,
        spaCy pipeline, and (with `corpora=True`) every persisted
        corpus's planner including its PPR matrix — so the FIRST query
        doesn't pay the cold start. Connector clients (Perplexity and
        friends) drop tool calls after ~5-15s; cold start alone can
        exceed that. Best-effort: a component that fails reports in the
        returned dict instead of raising, so a warmup problem never
        takes a server down with it."""
        report: dict[str, str] = {}
        try:
            self._get_embed()
            report["embedder"] = "ok"
        except Exception as e:
            report["embedder"] = f"failed: {e}"
        if self.nlp is None:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
                report["spacy"] = "ok"
            except Exception as e:
                report["spacy"] = f"failed: {e}"
        if corpora:
            for name in self.list():
                try:
                    self.planner(name)
                    report[f"corpus:{name}"] = "ok"
                except Exception as e:
                    report[f"corpus:{name}"] = f"failed: {e}"
        return report

    # ---- build / save ----------------------------------------------

    def build(
        self,
        name: str,
        documents: list[Document | dict | str],
        kg: EntityKG | None = None,
    ) -> dict:
        """Embed + index `documents`, build the entity KG (spaCy NER
        unless a prebuilt `kg` is passed), persist, and return stats.
        Rebuilding an existing name replaces it."""
        embed = self._get_embed()
        store = InMemoryStore.from_documents(documents, embed=embed)
        docs = store.get_many(store.all_ids())
        if kg is None:
            from .kg import build_entity_kg
            if self.nlp is None:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            kg = build_entity_kg(docs, nlp=self.nlp)

        cdir = self.root / name
        cdir.mkdir(parents=True, exist_ok=True)
        with open(cdir / "docs.jsonl", "w") as f:
            for d in docs:
                f.write(json.dumps({"id": d.id, "text": d.text,
                                    "metadata": d.metadata or {}}) + "\n")
        np.savez_compressed(
            cdir / "vectors.npz",
            doc_ids=np.array([d.id for d in docs]),
            vectors=np.stack([d.vector for d in docs]).astype(np.float32),
        )
        (cdir / "kg.json").write_text(json.dumps(kg_to_dict(kg)))
        meta = {
            "format_version": FORMAT_VERSION,
            "n_docs": len(docs),
            "kg_nodes": kg.graph.number_of_nodes(),
            "kg_edges": kg.graph.number_of_edges(),
        }
        (cdir / "meta.json").write_text(json.dumps(meta, indent=2))
        self._planners.pop(name, None)   # invalidate any cached planner
        return meta

    # ---- load / query ----------------------------------------------

    def load(self, name: str) -> tuple[InMemoryStore, EntityKG]:
        cdir = self.root / name
        if not (cdir / "meta.json").exists():
            raise FileNotFoundError(
                f"no corpus named {name!r} under {self.root}")
        # np.load's default forbids embedded objects — plain arrays only.
        npz = np.load(cdir / "vectors.npz")
        vecs = {i: v for i, v in zip(npz["doc_ids"], npz["vectors"])}
        docs = []
        with open(cdir / "docs.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                docs.append(Document(id=rec["id"], text=rec["text"],
                                     vector=vecs[rec["id"]],
                                     metadata=rec.get("metadata", {})))
        kg = kg_from_dict(json.loads((cdir / "kg.json").read_text()))
        return InMemoryStore(docs), kg

    def planner(self, name: str, config: PlannerConfig | None = None) -> Planner:
        """Planner for a named corpus, cached per manager (the PPR
        transition matrix is precomputed once at construction)."""
        if name not in self._planners:
            store, kg = self.load(name)
            self._planners[name] = Planner(
                store=store, kg=kg, nlp=self.nlp,
                embed=self._get_embed(), config=config,
            )
        return self._planners[name]

    def list(self) -> dict[str, dict]:
        if not self.root.exists():
            return {}
        out = {}
        for cdir in sorted(self.root.iterdir()):
            meta = cdir / "meta.json"
            if meta.exists():
                out[cdir.name] = json.loads(meta.read_text())
        return out
