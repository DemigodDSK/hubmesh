"""hubmesh — centrality-aware GraphRAG retrieval planner.

Public API:
    Planner       — main entry point
    Document      — input record (id, text, vector, metadata)
    RetrievalResult — what Planner.retrieve returns
"""
from .types import Document, RetrievalResult, ReasoningPath
from .planner import Planner

__all__ = ["Document", "RetrievalResult", "ReasoningPath", "Planner"]
__version__ = "0.0.1"
