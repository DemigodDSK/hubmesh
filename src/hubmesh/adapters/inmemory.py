"""Reference in-memory adapter.

Useful for tests, examples, and small-scale benchmarks. Production users will
swap in Pinecone/Qdrant/Weaviate/pgvector adapters (planned).
"""
from __future__ import annotations
from typing import Callable, Iterable
import numpy as np
from ..types import Document


class InMemoryStore:
    """Holds vectors and a precomputed k-NN proximity graph in memory."""

    def __init__(self, documents: list[Document], k: int = 10):
        if not documents:
            raise ValueError("InMemoryStore needs at least one document.")
        for d in documents:
            if d.vector is None:
                raise ValueError(f"Document {d.id} has no vector. "
                                 "Use InMemoryStore.from_documents(... embed=...)")
        self._docs: dict[str, Document] = {d.id: d for d in documents}
        self._ids: list[str] = [d.id for d in documents]
        self._id_to_idx: dict[str, int] = {i: k for k, i in enumerate(self._ids)}
        self._mat: np.ndarray = np.stack([d.vector for d in documents]).astype(np.float32)
        norms = np.linalg.norm(self._mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._mat_unit: np.ndarray = self._mat / norms
        self._k_graph = max(1, k)
        self._neighbor_cache: dict[str, list[str]] = {}
        self._build_knn_graph()

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document | str | dict],
        embed: Callable[[str], np.ndarray] | None = None,
        k: int = 10,
    ) -> "InMemoryStore":
        """Build a store from raw inputs. If items aren't already `Document`
        instances, embed them with `embed`."""
        docs: list[Document] = []
        for i, item in enumerate(documents):
            if isinstance(item, Document):
                docs.append(item)
                continue
            if isinstance(item, str):
                if embed is None:
                    raise ValueError("embed callable required when passing raw strings")
                docs.append(Document(id=str(i), text=item, vector=embed(item)))
                continue
            if isinstance(item, dict):
                d = Document(
                    id=str(item.get("id", i)),
                    text=item["text"],
                    vector=np.asarray(item["vector"]) if "vector" in item
                           else (embed(item["text"]) if embed else None),
                    metadata=item.get("metadata", {}),
                )
                if d.vector is None:
                    raise ValueError(f"Document {d.id} has no vector and no embed callable.")
                docs.append(d)
                continue
            raise TypeError(f"Unsupported document type: {type(item)}")
        return cls(documents=docs, k=k)

    def _build_knn_graph(self):
        n = len(self._ids)
        if n == 1:
            self._neighbor_cache[self._ids[0]] = []
            return
        sims = self._mat_unit @ self._mat_unit.T
        np.fill_diagonal(sims, -np.inf)
        k = min(self._k_graph, n - 1)
        topk = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        for i, row in enumerate(topk):
            order = row[np.argsort(-sims[i, row])]
            self._neighbor_cache[self._ids[i]] = [self._ids[j] for j in order]

    # ---- VectorStore protocol ----

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        q = np.asarray(query_vec, dtype=np.float32)
        qn = q / max(float(np.linalg.norm(q)), 1e-12)
        sims = self._mat_unit @ qn
        k = min(top_k, len(self._ids))
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self._ids[i], float(sims[i])) for i in idx]

    def get(self, doc_id: str) -> Document:
        return self._docs[doc_id]

    def get_many(self, doc_ids: list[str]) -> list[Document]:
        return [self._docs[i] for i in doc_ids]

    def neighbors(self, doc_id: str, k: int) -> list[str]:
        nb = self._neighbor_cache.get(doc_id, [])
        return nb[:k]

    def all_ids(self) -> list[str]:
        return list(self._ids)

    @property
    def dim(self) -> int:
        return self._mat.shape[1]

    # ---- internal helpers used by planner ----

    def vector_of(self, doc_id: str) -> np.ndarray:
        return self._mat[self._id_to_idx[doc_id]]
