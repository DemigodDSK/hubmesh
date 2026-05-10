"""Qdrant adapter — wraps a Qdrant collection as a hubmesh VectorStore.

Supports:
  • in-memory mode  (`QdrantClient(':memory:')`) for tests/demos
  • local mode      (`QdrantClient(path='./qdrant_data')`)
  • remote mode     (`QdrantClient(url='http://...:6333', api_key=...)`)

hubmesh's planner needs three things from a store:
  1. top-k cosine search          → Qdrant has it
  2. raw vector lookup by id      → Qdrant supports retrieve(with_vectors=True),
                                    but we cache locally for speed
  3. kNN proximity neighbours      → Qdrant doesn't expose its HNSW graph;
                                    we derive on demand via a per-id query
                                    (cached after first call)
"""
from __future__ import annotations
from typing import Iterable
import numpy as np

from ..types import Document


class QdrantStore:
    """Wraps a qdrant_client.QdrantClient. Documents are upserted into a
    named collection; metadata travels in the Qdrant payload."""

    def __init__(
        self,
        client,                          # qdrant_client.QdrantClient
        collection: str,
        dim: int,
        distance: str = "Cosine",        # Qdrant Distance enum string
        cache_vectors: bool = True,
    ):
        from qdrant_client.models import Distance, VectorParams

        self._client = client
        self.collection = collection
        self._dim = dim
        self._cache_vectors = cache_vectors
        self._vec_cache: dict[str, np.ndarray] = {}
        self._neighbor_cache: dict[str, list[str]] = {}
        self._all_ids_cache: list[str] | None = None

        # Create collection if missing
        existing = {c.name for c in client.get_collections().collections}
        if collection not in existing:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=dim, distance=getattr(Distance, distance.upper()),
                ),
            )

    # ---- ingest ----

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        client=None,
        collection: str = "hubmesh",
        path: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
        dim: int | None = None,
    ) -> "QdrantStore":
        """Build a store from already-vectorised Documents and upsert them."""
        from qdrant_client import QdrantClient

        if not documents:
            raise ValueError("QdrantStore.from_documents needs at least one document")
        for d in documents:
            if d.vector is None:
                raise ValueError(f"Document {d.id} has no vector")
        if dim is None:
            dim = int(documents[0].vector.shape[-1])

        if client is None:
            if url is not None:
                client = QdrantClient(url=url, api_key=api_key)
            elif path is not None:
                client = QdrantClient(path=path)
            else:
                client = QdrantClient(":memory:")

        store = cls(client, collection=collection, dim=dim)
        store.upsert(documents)
        return store

    def upsert(self, documents: list[Document], batch_size: int = 256):
        from qdrant_client.models import PointStruct
        # Qdrant point ids must be ints or UUIDs. We use a stable hash for
        # string ids and keep the original id in the payload.
        for i in range(0, len(documents), batch_size):
            chunk = documents[i:i + batch_size]
            points = []
            for d in chunk:
                pid = _stable_id(d.id)
                payload = {"hubmesh_id": d.id, "text": d.text}
                payload.update({f"meta_{k}": v for k, v in d.metadata.items()})
                vec = np.asarray(d.vector, dtype=np.float32)
                points.append(PointStruct(
                    id=pid, vector=vec.tolist(), payload=payload,
                ))
                if self._cache_vectors:
                    self._vec_cache[d.id] = vec
            self._client.upsert(collection_name=self.collection, points=points)
        self._all_ids_cache = None  # invalidate

    # ---- VectorStore protocol ----

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        q = np.asarray(query_vec, dtype=np.float32)
        hits = self._client.query_points(
            collection_name=self.collection,
            query=q.tolist(),
            limit=top_k,
            with_payload=True,
        ).points
        return [(h.payload["hubmesh_id"], float(h.score)) for h in hits]

    def get(self, doc_id: str) -> Document:
        from qdrant_client.models import HasIdCondition, Filter
        pid = _stable_id(doc_id)
        recs = self._client.retrieve(
            collection_name=self.collection,
            ids=[pid], with_payload=True, with_vectors=True,
        )
        if not recs:
            raise KeyError(doc_id)
        rec = recs[0]
        meta = {k.removeprefix("meta_"): v for k, v in (rec.payload or {}).items()
                if k.startswith("meta_")}
        vec = np.asarray(rec.vector, dtype=np.float32) if rec.vector is not None else None
        if vec is not None and self._cache_vectors:
            self._vec_cache[doc_id] = vec
        return Document(
            id=doc_id,
            text=(rec.payload or {}).get("text", ""),
            vector=vec,
            metadata=meta,
        )

    def get_many(self, doc_ids: list[str]) -> list[Document]:
        return [self.get(i) for i in doc_ids]

    def neighbors(self, doc_id: str, k: int) -> list[str]:
        if doc_id in self._neighbor_cache:
            return self._neighbor_cache[doc_id][:k]
        vec = self.vector_of(doc_id)
        # search returns the doc itself as the closest match — drop it
        hits = self.search(vec, top_k=k + 1)
        nbrs = [hid for hid, _ in hits if hid != doc_id][:k]
        self._neighbor_cache[doc_id] = nbrs
        return nbrs

    def all_ids(self) -> list[str]:
        if self._all_ids_cache is not None:
            return list(self._all_ids_cache)
        ids: list[str] = []
        offset = None
        while True:
            recs, offset = self._client.scroll(
                collection_name=self.collection,
                with_payload=True, with_vectors=False,
                limit=1024, offset=offset,
            )
            for r in recs:
                ids.append(r.payload["hubmesh_id"])
            if offset is None:
                break
        self._all_ids_cache = ids
        return list(ids)

    @property
    def dim(self) -> int:
        return self._dim

    # ---- internal helper used by Planner ----

    def vector_of(self, doc_id: str) -> np.ndarray:
        if self._cache_vectors and doc_id in self._vec_cache:
            return self._vec_cache[doc_id]
        recs = self._client.retrieve(
            collection_name=self.collection,
            ids=[_stable_id(doc_id)],
            with_payload=False, with_vectors=True,
        )
        if not recs or recs[0].vector is None:
            raise KeyError(doc_id)
        vec = np.asarray(recs[0].vector, dtype=np.float32)
        if self._cache_vectors:
            self._vec_cache[doc_id] = vec
        return vec


def _stable_id(doc_id: str) -> int:
    """Map a string id to a stable 63-bit int for Qdrant. Qdrant accepts
    int or UUID; we hash to int for compactness."""
    import hashlib
    h = hashlib.blake2b(doc_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") & ((1 << 63) - 1)
