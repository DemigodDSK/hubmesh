"""Vector store protocol — implement this to plug a new backend in."""
from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np
from ..types import Document


@runtime_checkable
class VectorStore(Protocol):
    """Minimal contract any backend (Pinecone, Qdrant, pgvector, ...) must satisfy.

    A store owns:
      • document storage   — id → Document
      • a vector index     — fast top-k by cosine similarity
      • optional pre-computed kNN graph (cheap to derive if absent)
    """

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Return [(doc_id, similarity)] of the top_k nearest documents."""

    def get(self, doc_id: str) -> Document:
        """Fetch a Document by id."""

    def get_many(self, doc_ids: list[str]) -> list[Document]:
        """Bulk fetch."""

    def neighbors(self, doc_id: str, k: int) -> list[str]:
        """Return up-to-k neighbor doc_ids in the proximity graph.
        Adapters that don't maintain a pre-built kNN graph should derive it
        from the vector index on demand."""

    def all_ids(self) -> list[str]:
        """Return every doc_id (used for full-graph operations during prototype;
        avoid calling on giant indices)."""

    @property
    def dim(self) -> int:
        """Embedding dimension."""
