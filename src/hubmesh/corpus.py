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

Persistence contract (scope verified by external review, 2026-09):
corpora are stored as immutable generation directories behind an
atomically-replaced CURRENT pointer. Writers are serialized by a
per-corpus flock (POSIX; on platforms without fcntl, concurrent
writers are unsupported). Readers resolve the pointer once and are
protected for `retention_seconds` from the moment their generation is
RETIRED — a reader that exceeds that grace period may fail loudly, and
never sees mixed generations. Power-loss durability (fsync ordering)
is not guaranteed.
"""
from __future__ import annotations
import json
import os
import re
import shutil
import threading
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import numpy as np
import networkx as nx

from .types import Document
from .adapters.inmemory import InMemoryStore
from .kg import EntityKG
from .planner import Planner, PlannerConfig

# v2 adds the "embedding" fingerprint block to meta.json. v1 corpora load
# with a warning; versions above the current one are rejected.
FORMAT_VERSION = 2
DEFAULT_ROOT = Path.home() / ".hubmesh" / "corpora"

# Corpus names are plain identifiers, never paths: must start with an
# alphanumeric (which also excludes the "." prefix used by temp/backup
# generations), then alphanumerics, "_", "-", ".". No separators, so a
# name can never traverse outside the configured root.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


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
    embed_identity: str | None = None   # e.g. "all-MiniLM-L6-v2"; recorded
                                        # in meta.json and checked on load
    retention_seconds: float = 900.0    # how long superseded generations
                                        # survive for in-flight readers
                                        # before GC (15 min default)
    _planners: dict = field(default_factory=dict, repr=False)
    _loaded: dict = field(default_factory=dict, repr=False)
    _embed_lock: threading.Lock = field(default_factory=threading.Lock,
                                        repr=False)
    _cache_lock: threading.RLock = field(default_factory=threading.RLock,
                                         repr=False)

    def __post_init__(self):
        self.root = Path(self.root).expanduser()

    # ---- name / path safety ----------------------------------------

    def _corpus_dir(self, name: str) -> Path:
        """Resolve a corpus name to its directory, rejecting anything that
        is not a plain identifier or that would escape the root (including
        via symlinked roots)."""
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise ValueError(
                f"invalid corpus name {name!r}: use 1-64 chars from "
                "[A-Za-z0-9._-], starting with a letter or digit "
                "(no path separators)")
        cdir = self.root / name
        root_res = self.root.resolve()
        if cdir.resolve().parent != root_res:
            raise ValueError(
                f"corpus name {name!r} resolves outside the corpus root")
        return cdir

    def _current_embed_identity(self) -> str | None:
        """Best-known identity of the active embedder WITHOUT loading it.
        Explicit `embed_identity` wins; otherwise, when the default
        lazy sentence-transformers path will be used, the model name is
        known from the environment. A custom callable with no declared
        identity yields None (compatibility then can't be verified)."""
        if self.embed_identity is not None:
            return self.embed_identity
        if self.embed is None:
            import os
            return os.environ.get("HUBMESH_EMBED_MODEL", "all-MiniLM-L6-v2")
        return None

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
                    # Initializing the default model must not lose its
                    # identity: once self.embed is set, the env-based
                    # inference in _current_embed_identity() no longer
                    # applies, so pin the resolved name now (an explicit
                    # user-provided identity is never overwritten).
                    if self.embed_identity is None:
                        self.embed_identity = model
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

        cdir = self._corpus_dir(name)
        vectors = np.stack([d.vector for d in docs]).astype(np.float32)
        meta = {
            "format_version": FORMAT_VERSION,
            "n_docs": len(docs),
            "kg_nodes": kg.graph.number_of_nodes(),
            "kg_edges": kg.graph.number_of_edges(),
            "embedding": {
                "identity": self._current_embed_identity(),
                "dim": int(vectors.shape[1]),
            },
        }

        # Generation-pointer publish. Layout: <root>/<name>/ holds
        # immutable generation dirs (gen-<hex8>/) plus a CURRENT pointer
        # file naming the live one. Publishing = write a complete new
        # generation, validate it, then atomically replace CURRENT
        # (os.replace of a file IS a transaction; two directory renames
        # are not — a crash between them can lose the live corpus, which
        # is exactly what this replaces). Readers resolve CURRENT once
        # and read only inside that generation, so a concurrent rebuild
        # can never hand them mixed text/vectors. The previous generation
        # is retained so in-flight readers stay valid across one rebuild;
        # older generations and stale staging dirs are pruned.
        cdir.mkdir(parents=True, exist_ok=True)
        gen_name = f"gen-{uuid.uuid4().hex[:8]}"

        # Stage -> publish -> prune runs under a per-corpus writer lock:
        # atomic pointer replacement alone does not serialize the
        # operations around it (an unlocked writer could prune with a
        # stale view and delete the generation another writer just
        # published). Readers never take the lock.
        with self._writer_lock(cdir):
            staging = cdir / (gen_name + ".staging")
            staging.mkdir(exist_ok=False)
            try:
                with open(staging / "docs.jsonl", "w") as f:
                    for d in docs:
                        f.write(json.dumps({"id": d.id, "text": d.text,
                                            "metadata": d.metadata or {}})
                                + "\n")
                np.savez_compressed(
                    staging / "vectors.npz",
                    doc_ids=np.array([d.id for d in docs]),
                    vectors=vectors,
                )
                (staging / "kg.json").write_text(json.dumps(kg_to_dict(kg)))
                (staging / "meta.json").write_text(json.dumps(meta, indent=2))
                # validate the generation before it can become CURRENT
                check = np.load(staging / "vectors.npz")
                if len(check["doc_ids"]) != meta["n_docs"]:
                    raise IOError("corpus generation failed self-validation")
                json.loads((staging / "meta.json").read_text())
                staging.rename(cdir / gen_name)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise

            prev = self._read_pointer(cdir)
            ptr_tmp = cdir / f"CURRENT.tmp-{gen_name}"   # writer-unique
            ptr_tmp.write_text(gen_name)
            os.replace(ptr_tmp, cdir / "CURRENT")        # atomic publish
            # Retirement bookkeeping: the reader-protection window starts
            # when a generation STOPS BEING CURRENT, not when it was
            # created — an old live corpus's first rebuild must still give
            # its in-flight readers the full grace period.
            if prev and prev != gen_name and (cdir / prev).exists():
                self._mark_retired(cdir / prev)
            self._prune_locked(cdir, current=gen_name)

        with self._cache_lock:
            self._drop_name(name)    # every cached planner/config for this name
        return meta

    # ---- writer coordination / retention ---------------------------

    @contextmanager
    def _writer_lock(self, cdir: Path):
        """Advisory per-corpus exclusive lock (flock on <corpus>/.lock).
        Serializes stage/publish/prune across writers — threads and
        processes on the same host. Readers are lock-free by design. On
        platforms without fcntl (Windows), degrades to no locking;
        concurrent writers there are unsupported."""
        lock_path = cdir / ".lock"
        try:
            import fcntl
        except ImportError:              # non-POSIX: best effort
            yield
            return
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    @staticmethod
    def _mark_retired(gen_dir: Path) -> None:
        """Record the moment a generation stopped being current. The
        marker lives inside the generation dir so it is deleted with it
        and never orphans bookkeeping."""
        import time
        try:
            (gen_dir / ".retired").write_text(repr(time.time()))
        except OSError:
            pass   # conservative path in _prune_locked covers a miss

    def _prune_locked(self, cdir: Path, current: str) -> None:
        """Garbage-collect under the writer lock.

        Deletion policy (round-4 review): the grace period runs from
        RETIREMENT, not creation — a generation that was live for days
        still gets the full `retention_seconds` after being replaced.
        A non-current generation with no readable retirement marker
        (crash between publish and bookkeeping, or a pre-fix layout) is
        handled conservatively: the marker is written NOW and the
        generation is skipped this pass, so its window starts fresh
        rather than being treated as already expired.
        Orphaned staging dirs and pointer temp files are safe to remove
        while the lock is held (live writers hold it during staging).
        Legacy flat-layout files are NEVER deleted: a legacy reader's
        lifetime is unknowable, and superseded flat files are inert
        once CURRENT exists."""
        import time
        now = time.time()
        for child in cdir.iterdir():
            name = child.name
            try:
                if name.endswith(".staging") or name.startswith("CURRENT.tmp-"):
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
                elif name.startswith("gen-") and name != current:
                    retired_at = None
                    try:
                        retired_at = float(
                            (child / ".retired").read_text().strip())
                    except (OSError, ValueError):
                        pass
                    if retired_at is None:
                        self._mark_retired(child)   # clock starts now
                        continue
                    if now - retired_at > self.retention_seconds:
                        shutil.rmtree(child, ignore_errors=True)
            except OSError:
                continue   # GC is best-effort; never fail a publish over it

    @staticmethod
    def _read_pointer(cdir: Path) -> str | None:
        """Name of the live generation, or None (missing/legacy layout)."""
        cur = cdir / "CURRENT"
        if not cur.exists():
            return None
        gen = cur.read_text().strip()
        if not re.fullmatch(r"gen-[0-9a-f]{8}", gen):
            raise ValueError(
                f"corrupt CURRENT pointer in {cdir}: {gen!r}")
        return gen

    # ---- load / query ----------------------------------------------

    def load(self, name: str) -> tuple[InMemoryStore, EntityKG]:
        """Load the live generation of `name` as (store, kg)."""
        _, store, kg = self._load_gen(name)
        return store, kg

    def _load_gen(self, name: str) -> tuple[str | None, InMemoryStore, EntityKG]:
        """`load()` plus the generation name the data came from (None for
        the legacy flat layout) — the cache key for freshness checks."""
        cdir = self._corpus_dir(name)
        # Resolve the live generation ONCE; every subsequent read stays
        # inside it, so a rebuild that publishes mid-load cannot hand this
        # reader text from one generation and vectors from another.
        gen = self._read_pointer(cdir) if cdir.exists() else None
        gdir = (cdir / gen) if gen else cdir     # None -> legacy flat layout
        if gen and not gdir.exists():
            raise FileNotFoundError(
                f"corpus {name!r}: CURRENT names generation {gen} but its "
                "directory is missing — pruned during an unlocked write or "
                "externally deleted; rebuild the corpus")
        if not (gdir / "meta.json").exists():
            raise FileNotFoundError(
                f"no corpus named {name!r} under {self.root}")
        meta = json.loads((gdir / "meta.json").read_text())
        stored_version = meta.get("format_version", 0)
        if stored_version > FORMAT_VERSION:
            raise ValueError(
                f"corpus {name!r} has format_version {stored_version}, "
                f"newer than this hubmesh ({FORMAT_VERSION}) — upgrade "
                "hubmesh to read it")
        emb = meta.get("embedding")
        if emb is None:
            warnings.warn(
                f"corpus {name!r} predates embedding fingerprints "
                "(format v1); compatibility with the current embedder "
                "cannot be verified — rebuild to record one",
                stacklevel=2)
        else:
            current = self._current_embed_identity()
            stored = emb.get("identity")
            if stored and current and stored != current:
                raise ValueError(
                    f"corpus {name!r} was embedded with {stored!r} but the "
                    f"current embedder is {current!r}. Same-dimension "
                    "mismatches corrupt retrieval silently. Fix: set "
                    f"HUBMESH_EMBED_MODEL={stored} (or pass embed_identity/"
                    "a matching embed callable), or rebuild the corpus")
            if stored and current is None:
                warnings.warn(
                    f"corpus {name!r} was embedded with {stored!r}; the "
                    "current custom embedder declares no embed_identity, so "
                    "compatibility cannot be verified", stacklevel=2)
        # np.load's default forbids embedded objects — plain arrays only.
        npz = np.load(gdir / "vectors.npz")
        if emb is not None and emb.get("dim") is not None:
            actual_dim = int(npz["vectors"].shape[1])
            if actual_dim != int(emb["dim"]):
                raise ValueError(
                    f"corpus {name!r}: meta.json declares embedding dim "
                    f"{emb['dim']} but vectors.npz has dim {actual_dim} — "
                    "the corpus files are inconsistent; rebuild it")
        vecs = {i: v for i, v in zip(npz["doc_ids"], npz["vectors"])}
        docs = []
        with open(gdir / "docs.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                docs.append(Document(id=rec["id"], text=rec["text"],
                                     vector=vecs[rec["id"]],
                                     metadata=rec.get("metadata", {})))
        kg = kg_from_dict(json.loads((gdir / "kg.json").read_text()))
        return gen, InMemoryStore(docs), kg

    def _peek_gen(self, name: str) -> str | None:
        """The generation CURRENT names right now (None: legacy layout or
        no corpus). One small file read — cheap enough per request."""
        cdir = self._corpus_dir(name)
        return self._read_pointer(cdir) if cdir.exists() else None

    def _drop_name(self, name: str) -> None:
        """Forget every cached object derived from `name` (caller holds
        the cache lock)."""
        self._loaded.pop(name, None)
        for key in [k for k in self._planners if k[0] == name]:
            self._planners.pop(key, None)

    def _load_current(self, name: str) -> tuple[str | None, InMemoryStore, EntityKG]:
        """Load the live generation and confirm it is STILL live after the
        load. A publish that landed mid-load triggers a reload (bounded);
        exhausting the bound returns the last load, which the caller then
        serves without pinning it."""
        gen = store = kg = None
        for _ in range(3):
            gen, store, kg = self._load_gen(name)
            if self._peek_gen(name) == gen:
                break
        return gen, store, kg

    @staticmethod
    def _config_key(config: PlannerConfig | None) -> str:
        """Stable identity of a PlannerConfig for cache keying — the cache
        must never hand back a planner built for a different config
        (external review: convergence-off then convergence-on returned the
        first planner unchanged)."""
        if config is None:
            return "default"
        from dataclasses import asdict
        return json.dumps(asdict(config), sort_keys=True, default=str)

    def planner(self, name: str, config: PlannerConfig | None = None) -> Planner:
        """Planner for a named corpus, cached per manager AND per config
        (the PPR transition matrix is precomputed once at construction;
        store+KG are loaded once and shared across configs).

        Freshness contract (external review, 2026-09-14): the cache is
        keyed by GENERATION. Every call re-reads the CURRENT pointer, so
        a rebuild published by this manager or by another process is
        picked up on the next call, and a load that raced with a publish
        is never installed over the newer generation — the retired one
        is served at most to the request that was already loading it,
        never pinned.
        """
        key = (name, self._config_key(config))
        current = self._peek_gen(name)
        with self._cache_lock:
            loaded = self._loaded.get(name)
            if loaded is not None and loaded[0] != current:
                self._drop_name(name)           # superseded generation
                loaded = None
            if loaded is not None and key in self._planners:
                return self._planners[key]
        if loaded is None:
            loaded = self._load_current(name)   # slow path, outside the lock
        gen, store, kg = loaded
        planner = Planner(store=store, kg=kg, nlp=self.nlp,
                          embed=self._get_embed(), config=config)
        with self._cache_lock:
            # Install only if CURRENT still names the generation this
            # planner was built from; otherwise serve it uncached and let
            # the next call reload.
            if self._peek_gen(name) == gen:
                existing = self._loaded.get(name)
                if existing is None or existing[0] != gen:
                    self._drop_name(name)
                    self._loaded[name] = loaded
                self._planners[key] = planner
        return planner

    def list(self) -> dict[str, dict]:
        if not self.root.exists():
            return {}
        out = {}
        for cdir in sorted(self.root.iterdir()):
            if cdir.name.startswith("."):
                continue
            try:
                gen = self._read_pointer(cdir)
            except ValueError:
                continue   # corrupt pointer — not listable
            meta = (cdir / gen / "meta.json") if gen else cdir / "meta.json"
            if meta.exists():
                out[cdir.name] = json.loads(meta.read_text())
        return out
