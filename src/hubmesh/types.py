"""Public data types."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class Document:
    """Input record. `vector` may be None if it'll be embedded by the store."""
    id: str
    text: str
    vector: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredDocument:
    """A document with retrieval-time scores attached."""
    doc: Document
    similarity: float           # cosine similarity to query
    ppr_score: float            # personalized PageRank score
    composite_score: float      # the multi-component output score
    rank: int


@dataclass
class ReasoningPath:
    """A multi-hop path through the induced subgraph."""
    node_ids: list[str]
    edges: list[tuple[str, str]]
    score: float


@dataclass
class RetrievalResult:
    """Returned by Planner.retrieve."""
    query: str
    context: str                          # packed context for an LLM
    sources: list[ScoredDocument]         # ranked, with scores
    reasoning: list[ReasoningPath]        # multi-hop paths (may be empty)
    debug: dict[str, Any] = field(default_factory=dict)
