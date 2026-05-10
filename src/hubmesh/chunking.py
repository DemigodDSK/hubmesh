"""Document chunking — split long source documents into retrievable units.

Production users typically index *passages*, not whole documents. A
1000-page PDF retrieved as a single chunk is useless; chunked into ~200-
token paragraphs it becomes effective.

Two strategies are exposed:

  • `chunk_by_sentences`  — split on sentence boundaries with a soft
                            target token budget. Sentence-respecting,
                            no truncation mid-sentence. Best for prose.
  • `chunk_by_chars`      — fixed-size character window with overlap.
                            Fast, language-agnostic, may cut sentences.
                            Best for code/markup or when you don't have
                            an NLP pipeline.

Both return `Document` objects ready for ingestion. The original
document id is preserved as `metadata['source_id']`; chunk ids are
`{source_id}#chunk{i}`.
"""
from __future__ import annotations
import re
from typing import Iterable
from .types import Document


_TOKEN_RATIO = 4   # ~4 chars per token (rough English approximation)


def chunk_by_chars(
    source_id: str,
    text: str,
    chunk_chars: int = 800,
    overlap_chars: int = 100,
    metadata: dict | None = None,
) -> list[Document]:
    """Fixed-size character window. Fast, language-agnostic.

    Defaults: 800 chars (~200 tokens) with 100-char overlap.
    """
    if not text:
        return []
    metadata = dict(metadata or {})
    metadata["source_id"] = source_id
    n = len(text)
    # If the source fits in one chunk, short-circuit.
    if n <= chunk_chars:
        return [Document(
            id=f"{source_id}#chunk0", text=text.strip(), vector=None,
            metadata={**metadata, "chunk_idx": 0,
                      "char_start": 0, "char_end": n},
        )]
    # Sane bounds: overlap must leave forward progress.
    if overlap_chars >= chunk_chars:
        overlap_chars = max(0, chunk_chars // 4)
    step = chunk_chars - overlap_chars
    out: list[Document] = []
    i = 0
    chunk_idx = 0
    while i < n:
        chunk = text[i:i + chunk_chars].strip()
        if chunk:
            out.append(Document(
                id=f"{source_id}#chunk{chunk_idx}",
                text=chunk, vector=None,
                metadata={**metadata, "chunk_idx": chunk_idx,
                          "char_start": i,
                          "char_end": min(i + chunk_chars, n)},
            ))
            chunk_idx += 1
        i += step
    return out


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def _approx_sentences(text: str) -> list[str]:
    """A quick sentence splitter that handles the common cases.
    For higher fidelity pass nlp= a spaCy pipeline (see chunk_by_sentences)."""
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def chunk_by_sentences(
    source_id: str,
    text: str,
    target_tokens: int = 200,
    overlap_sentences: int = 1,
    metadata: dict | None = None,
    nlp=None,
) -> list[Document]:
    """Sentence-respecting chunker. Greedily packs sentences into chunks
    until each chunk approaches `target_tokens` (estimated as chars/4).

    Pass `nlp` (spaCy pipeline) for higher-quality sentence splitting on
    tricky text (abbreviations, quotations, etc.). Without it, a regex
    fallback is used.
    """
    if not text:
        return []
    metadata = dict(metadata or {})
    metadata["source_id"] = source_id

    if nlp is not None:
        sents = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    else:
        sents = _approx_sentences(text)
    if not sents:
        return []

    target_chars = target_tokens * _TOKEN_RATIO
    out: list[Document] = []
    chunk_idx = 0
    i = 0
    while i < len(sents):
        cur_chars = 0
        end = i
        while end < len(sents) and cur_chars + len(sents[end]) + 1 <= target_chars:
            cur_chars += len(sents[end]) + 1
            end += 1
        # Always include at least one sentence (avoids infinite loop on a
        # single sentence longer than the target)
        if end == i:
            end = i + 1
        chunk_text = " ".join(sents[i:end])
        out.append(Document(
            id=f"{source_id}#chunk{chunk_idx}",
            text=chunk_text,
            vector=None,
            metadata={**metadata, "chunk_idx": chunk_idx,
                      "sent_start": i, "sent_end": end},
        ))
        chunk_idx += 1
        # advance, with overlap
        i = max(end - overlap_sentences, i + 1)
    return out


def chunk_documents(
    sources: Iterable,            # iterable of (id, text) or Document with text
    strategy: str = "sentences",
    **kwargs,
) -> list[Document]:
    """Bulk-chunk a corpus. Yields Documents ready for embedding."""
    fn = chunk_by_sentences if strategy == "sentences" else chunk_by_chars
    out: list[Document] = []
    for src in sources:
        if isinstance(src, Document):
            sid, text, md = src.id, src.text, src.metadata
        elif isinstance(src, tuple) and len(src) == 2:
            sid, text = src
            md = {}
        elif isinstance(src, dict):
            sid, text, md = src["id"], src["text"], src.get("metadata", {})
        else:
            raise TypeError(f"Unsupported source type: {type(src)}")
        out.extend(fn(sid, text, metadata=md, **kwargs))
    return out
