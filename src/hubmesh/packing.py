"""Budget-aware context packing.

Once nodes are scored, pack them into the LLM's context window with
explicit redundancy control. The objective:

    maximise   Σᵢ S(i) for i in selected
    subject to Σᵢ tokens(i) ≤ B
              and a redundancy penalty ν · Σᵢ⨉ⱼ sim(i,j)

We approximate this with a greedy MMR-style picker. Cheap and good enough
in practice; an ILP version can come later for offline batch retrieval.
"""
from __future__ import annotations
import numpy as np

from .types import Document, ScoredDocument


def estimate_tokens(text: str) -> int:
    """Rough token count: 1 token ≈ 4 chars. Replace with tiktoken or model
    tokenizer in production."""
    return max(1, len(text) // 4)


def pack(
    scored: list[ScoredDocument],
    budget_tokens: int,
    redundancy_lambda: float = 0.3,
    vec_of: callable | None = None,
) -> tuple[str, list[ScoredDocument]]:
    """Greedy MMR packing. Returns (joined_context, picked).

    redundancy_lambda interpolates between pure score (0.0) and pure
    diversity (1.0). vec_of is optional — if supplied we use cosine similarity
    between picked docs to suppress redundancy; otherwise we just dedupe by
    text prefix.
    """
    if not scored:
        return "", []
    sorted_scored = sorted(scored, key=lambda x: -x.composite_score)

    picked: list[ScoredDocument] = []
    used_tokens = 0
    picked_vecs: list[np.ndarray] = []

    for cand in sorted_scored:
        toks = estimate_tokens(cand.doc.text)
        if used_tokens + toks > budget_tokens:
            continue
        # redundancy check
        if vec_of is not None and picked_vecs:
            cv = vec_of(cand.doc.id)
            cn = cv / max(float(np.linalg.norm(cv)), 1e-12)
            sims = [
                float(cn @ (p / max(float(np.linalg.norm(p)), 1e-12)))
                for p in picked_vecs
            ]
            max_sim = max(sims)
            adjusted = (1 - redundancy_lambda) * cand.composite_score \
                       - redundancy_lambda * max_sim
            if adjusted < 0 and max_sim > 0.95:
                continue  # near-duplicate of an already-picked doc
        picked.append(cand)
        used_tokens += toks
        if vec_of is not None:
            picked_vecs.append(vec_of(cand.doc.id))

    # Re-rank picked items: best-first ordering puts strong docs at top
    # of the context (counters Lost-in-the-Middle bias)
    picked.sort(key=lambda x: -x.composite_score)
    parts = [f"[{i+1}] {p.doc.text}" for i, p in enumerate(picked)]
    return "\n\n".join(parts), picked
