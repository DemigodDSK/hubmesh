"""hubmesh — centrality-aware GraphRAG retrieval planner.

Public API:
    Planner            — main entry point
    Document           — input record (id, text, vector, metadata)
    RetrievalResult    — what Planner.retrieve returns
    ReasoningPath      — multi-hop trace returned alongside results
    chunk_by_sentences,
    chunk_by_chars,
    chunk_documents    — helpers for splitting long source documents
"""
from .types import Document, RetrievalResult, ReasoningPath, ScoredDocument
from .planner import Planner, PlannerConfig
from .chunking import chunk_by_sentences, chunk_by_chars, chunk_documents

__all__ = [
    "Document", "RetrievalResult", "ReasoningPath", "ScoredDocument",
    "Planner", "PlannerConfig",
    "chunk_by_sentences", "chunk_by_chars", "chunk_documents",
]
__version__ = "0.1.0"
