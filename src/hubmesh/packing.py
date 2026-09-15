"""Budget-aware context packing.

Once nodes are scored, pack them into the LLM's context window with a
document cap and a near-duplicate filter. The selection policy is
greedy score-first: candidates are visited best-first, a candidate is
skipped when it would exceed the token budget or when it is a
near-duplicate (cosine > 0.95) of something already picked. This is NOT
an iterative MMR re-selection — it never re-ranks remaining candidates
by diversity — and is documented as such.

Token accounting measures the COMPLETE serialized context (numbering
prefixes and separators included) that a candidate would produce, with
the same counter used for the budget. Counters are not additive —
chars/4 rounds per piece, BPE tokenizers merge across boundaries — so
summing per-piece counts can exceed the budget on the joined string
(external review, 2026-09-14). Measuring the exact string that would be
returned is what makes `count_tokens(context) <= budget_tokens` hold
by construction. Pass `count_tokens=` (e.g. a tiktoken-backed callable)
for model-exact budgets; the default is a chars/4 estimate.
"""
from __future__ import annotations
from typing import Callable
import numpy as np

from .types import Document, ScoredDocument


def estimate_tokens(text: str) -> int:
    """Rough token count: 1 token ≈ 4 chars. Replace with tiktoken or model
    tokenizer in production via pack(count_tokens=...)."""
    return max(1, len(text) // 4)


_SEP = "\n\n"


def pack(
    scored: list[ScoredDocument],
    budget_tokens: int,
    redundancy_lambda: float = 0.3,
    vec_of: Callable | None = None,
    max_docs: int | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> tuple[str, list[ScoredDocument]]:
    """Greedy score-first packing with a near-duplicate filter.
    Returns (joined_context, picked); the context is built from exactly
    the returned `picked` list, so provenance and text always agree.

    `max_docs` caps how many documents enter the context — pass the
    caller's top_k so the context never contains more documents than the
    sources it reports. `count_tokens` measures the serialized pieces
    (prefix + separator + body) against `budget_tokens`.
    """
    if not scored:
        return "", []
    count = count_tokens or estimate_tokens
    # Deterministic order: score desc, then doc id — equal scores must
    # not depend on input order or hash seeds.
    sorted_scored = sorted(scored,
                           key=lambda x: (-x.composite_score, x.doc.id))

    picked: list[ScoredDocument] = []
    parts: list[str] = []            # serialized pieces, in output order
    picked_vecs: list[np.ndarray] = []

    for cand in sorted_scored:
        if max_docs is not None and len(picked) >= max_docs:
            break
        # Budget check on the COMPLETE context this document would
        # produce — the exact string returned below if it is accepted —
        # never on a sum of per-piece counts (counters are not additive).
        trial = parts + [f"[{len(parts) + 1}] {cand.doc.text}"]
        if count(_SEP.join(trial)) > budget_tokens:
            continue
        if vec_of is not None and picked_vecs:
            cv = vec_of(cand.doc.id)
            cn = cv / max(float(np.linalg.norm(cv)), 1e-12)
            max_sim = max(
                float(cn @ (p / max(float(np.linalg.norm(p)), 1e-12)))
                for p in picked_vecs
            )
            adjusted = (1 - redundancy_lambda) * cand.composite_score \
                       - redundancy_lambda * max_sim
            if adjusted < 0 and max_sim > 0.95:
                continue  # near-duplicate of an already-picked doc
        picked.append(cand)
        parts = trial
        if vec_of is not None:
            picked_vecs.append(vec_of(cand.doc.id))

    # Best-first ordering in the context (counters Lost-in-the-Middle
    # bias); `parts` was built in that order, and is exactly what was
    # measured against the budget.
    return _SEP.join(parts), picked
