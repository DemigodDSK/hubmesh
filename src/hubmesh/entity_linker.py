"""Entity linkers — pluggable strategies for canonicalising entity mentions.

The default linker (substring-collapse in `kg.py`) is fast but fragile.
Different forms of the same entity ("Bill Clinton" / "Clinton" /
"President Clinton") sometimes get unified, sometimes don't, depending
on which substring relations exist in the corpus.

This module adds an **embedding-based linker** that:

  1. Encodes each canonical mention with a sentence-transformer (or any
     callable text → np.ndarray)
  2. Greedily clusters mentions by cosine similarity over a configurable
     threshold
  3. Picks the longest mention in each cluster as the canonical form

It plugs into `kg.build_entity_kg` via the `linker=` argument: when
provided, the substring-collapse step is replaced by clustering.

Other linkers (BLINK, LLM-based, learned) can be added with the same
interface — see `Linker` protocol below.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol
import numpy as np


class Linker(Protocol):
    """A linker maps a set of raw mentions to a canonical-form mapping.

    Returns a dict {raw_mention -> canonical_form} where every distinct
    canonical form represents one logical entity. The downstream KG
    construction uses canonical forms as node ids.
    """
    def link(self, mentions: Iterable[str]) -> dict[str, str]: ...


@dataclass
class SubstringLinker:
    """The default behaviour from `kg.canonicalize` + substring collapse.
    Reified as a Linker so it's swappable. Cheap and dependency-free."""
    def link(self, mentions: Iterable[str]) -> dict[str, str]:
        from .kg import canonicalize
        canons = {m: canonicalize(m) for m in mentions if canonicalize(m)}
        # Collapse: longer canonical absorbs shorter token-aligned substrings
        unique = sorted(set(canons.values()), key=len, reverse=True)
        kept: list[str] = []
        absorbed: dict[str, str] = {}   # short → long
        for c in unique:
            target = None
            for k in kept:
                if c == k:
                    continue
                if (f" {c} " in f" {k} "
                        or k.startswith(c + " ")
                        or k.endswith(" " + c)):
                    target = k
                    break
            if target is None:
                kept.append(c)
            else:
                absorbed[c] = target
        out = {}
        for m, c in canons.items():
            out[m] = absorbed.get(c, c)
        return out


@dataclass
class EmbeddingLinker:
    """Cluster canonicalised mentions by cosine similarity over their
    embeddings. Mentions whose unit-vector dot product exceeds
    `threshold` are merged into one canonical form (the longest mention
    in the cluster).

    This is a poor-man's entity linker — it doesn't use external
    knowledge, doesn't handle ambiguity ("Apple" the company vs the
    fruit), and won't recover entities whose surface forms are
    completely different ("US" / "the States"). For Wikipedia-style
    corpora it still meaningfully improves coverage over substring-only
    canonicalisation.
    """
    embed: Callable[[list[str]], np.ndarray]   # batched embedder
    threshold: float = 0.82
    min_canonical_chars: int = 2

    def link(self, mentions: Iterable[str]) -> dict[str, str]:
        from .kg import canonicalize
        # First pass: canonicalize and dedupe
        raw_to_canon = {m: canonicalize(m) for m in mentions if canonicalize(m)}
        canons = sorted(set(raw_to_canon.values()))
        if len(canons) <= 1:
            return raw_to_canon

        vecs = np.asarray(self.embed(canons), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = vecs / norms

        # Greedy clustering: assign each canonical to an existing cluster if
        # its cosine to that cluster's representative ≥ threshold; else
        # start a new cluster. Representatives are the *longest* mention so
        # short forms get absorbed into their longer counterparts.
        order = sorted(range(len(canons)), key=lambda i: -len(canons[i]))
        cluster_reps: list[int] = []
        canon_to_rep_idx: dict[int, int] = {}
        for i in order:
            assigned = False
            for r in cluster_reps:
                sim = float(unit[i] @ unit[r])
                if sim >= self.threshold:
                    canon_to_rep_idx[i] = r
                    assigned = True
                    break
            if not assigned:
                cluster_reps.append(i)
                canon_to_rep_idx[i] = i

        canon_to_canonical = {
            canons[i]: canons[canon_to_rep_idx[i]]
            for i in range(len(canons))
        }
        return {m: canon_to_canonical[c] for m, c in raw_to_canon.items()}


def make_st_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Convenience: returns a callable that batched-encodes text via
    sentence-transformers, normalising automatically. Importing
    sentence-transformers is deferred to call time so users without it
    can still use SubstringLinker."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    def _embed(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float32)

    return _embed
