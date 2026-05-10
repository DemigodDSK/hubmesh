"""Vector store adapters."""
from .base import VectorStore
from .inmemory import InMemoryStore

__all__ = ["VectorStore", "InMemoryStore"]

# Qdrant is an optional dependency — only export if installed.
try:
    from .qdrant import QdrantStore  # noqa: F401
    __all__.append("QdrantStore")
except ImportError:
    pass

# Chroma is also optional.
try:
    from .chroma import ChromaStore  # noqa: F401
    __all__.append("ChromaStore")
except ImportError:
    pass
