"""Chroma adapter — wraps a Chroma collection as a hubmesh VectorStore.

Supports:
  • ephemeral mode  (`chromadb.EphemeralClient()`) for tests/demos
  • persistent mode (`chromadb.PersistentClient(path=...)`) for local
  • HTTP mode       (`chromadb.HttpClient(host=..., port=...)`) for remote

The Chroma client allows arbitrary string ids natively (unlike Qdrant
which requires int/UUID), so this adapter is simpler than the Qdrant one.
"""
from __future__ import annotations
from typing import Iterable
import numpy as np

from ..types import Document


class ChromaStore:
    """Wraps chromadb.Client + a single Collection."""

    def __init__(
        self,
        client,                               # chromadb.api.client.Client
        collection,                           # chromadb Collection
        cache_vectors: bool = True,
    ):
        self._client = client
        self._collection = collection
        self._cache_vectors = cache_vectors
        self._vec_cache: dict[str, np.ndarray] = {}
        self._neighbor_cache: dict[str, list[str]] = {}
        self._dim_cache: int | None = None

    # ---- ingest ----

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        client=None,
        collection_name: str = "hubmesh",
        persist_directory: str | None = None,
        host: str | None = None,
        port: int | None = None,
    ) -> "ChromaStore":
        import chromadb

        if not documents:
            raise ValueError("ChromaStore.from_documents needs at least one document")
        for d in documents:
            if d.vector is None:
                raise ValueError(f"Document {d.id} has no vector")

        if client is None:
            if host is not None:
                client = chromadb.HttpClient(host=host, port=port or 8000)
            elif persist_directory is not None:
                client = chromadb.PersistentClient(path=persist_directory)
            else:
                client = chromadb.EphemeralClient()

        # Get or create the collection. We use cosine distance (Chroma's
        # default for `hnsw:space=cosine` configurations).
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        store = cls(client, collection)
        store.upsert(documents)
        return store

    def upsert(self, documents: list[Document], batch_size: int = 256):
        for i in range(0, len(documents), batch_size):
            chunk = documents[i:i + batch_size]
            ids = [d.id for d in chunk]
            embeddings = [np.asarray(d.vector, dtype=np.float32).tolist()
                          for d in chunk]
            metadatas = [{"text": d.text, **d.metadata} for d in chunk]
            documents_text = [d.text for d in chunk]
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_text,
            )
            if self._cache_vectors:
                for d in chunk:
                    self._vec_cache[d.id] = np.asarray(d.vector, dtype=np.float32)

    # ---- VectorStore protocol ----

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        q = np.asarray(query_vec, dtype=np.float32).tolist()
        result = self._collection.query(
            query_embeddings=[q], n_results=top_k,
            include=["distances"],
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        # Chroma cosine "distance" = 1 - cosine_similarity.
        # We expose similarity (higher = better) for consistency.
        return [(ids[i], 1.0 - float(distances[i])) for i in range(len(ids))]

    def get(self, doc_id: str) -> Document:
        rec = self._collection.get(
            ids=[doc_id],
            include=["embeddings", "metadatas", "documents"],
        )
        if not rec["ids"]:
            raise KeyError(doc_id)
        meta_raw = rec["metadatas"][0] or {}
        text = meta_raw.pop("text", rec["documents"][0] if rec["documents"] else "")
        vec = rec["embeddings"][0] if rec["embeddings"] is not None else None
        if vec is not None:
            vec = np.asarray(vec, dtype=np.float32)
            if self._cache_vectors:
                self._vec_cache[doc_id] = vec
        return Document(id=doc_id, text=text, vector=vec, metadata=meta_raw)

    def get_many(self, doc_ids: list[str]) -> list[Document]:
        return [self.get(i) for i in doc_ids]

    def neighbors(self, doc_id: str, k: int) -> list[str]:
        if doc_id in self._neighbor_cache:
            return self._neighbor_cache[doc_id][:k]
        vec = self.vector_of(doc_id)
        hits = self.search(vec, top_k=k + 1)
        nbrs = [hid for hid, _ in hits if hid != doc_id][:k]
        self._neighbor_cache[doc_id] = nbrs
        return nbrs

    def all_ids(self) -> list[str]:
        rec = self._collection.get(include=[])
        return list(rec["ids"])

    @property
    def dim(self) -> int:
        if self._dim_cache is not None:
            return self._dim_cache
        # Look up any cached vector or fetch one
        if self._vec_cache:
            self._dim_cache = next(iter(self._vec_cache.values())).shape[0]
            return self._dim_cache
        ids = self.all_ids()
        if not ids:
            raise RuntimeError("Empty collection — no dim to infer")
        v = self.vector_of(ids[0])
        self._dim_cache = int(v.shape[0])
        return self._dim_cache

    # ---- internal helper ----

    def vector_of(self, doc_id: str) -> np.ndarray:
        if self._cache_vectors and doc_id in self._vec_cache:
            return self._vec_cache[doc_id]
        rec = self._collection.get(ids=[doc_id], include=["embeddings"])
        if not rec["ids"] or rec["embeddings"] is None or not len(rec["embeddings"]):
            raise KeyError(doc_id)
        vec = np.asarray(rec["embeddings"][0], dtype=np.float32)
        if self._cache_vectors:
            self._vec_cache[doc_id] = vec
        return vec
