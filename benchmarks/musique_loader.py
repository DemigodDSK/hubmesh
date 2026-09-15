"""MuSiQue loader for hubmesh benchmarks.

MuSiQue (Trivedi et al., 2022) provides 2-, 3-, and 4-hop questions with
explicit decomposition. Each question has 20 candidate paragraphs (more
distractors than HotpotQA's 10), and `is_supporting` is annotated per
paragraph.

We mirror the HotpotQA loader's interface so the benchmark runner is
shared. Paragraphs are pooled across questions and deduped by title.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class MuSiQueExample:
    qid: str
    question: str
    answer: str
    n_hops: int
    gold_titles: list[str]
    candidate_titles: list[str]
    candidate_paragraphs: list[tuple[str, str]]


def _is_supporting(p: dict) -> bool:
    """The HF dataset stores `is_supporting` as the string "True"/"False".
    Be tolerant of either bool or str."""
    v = p.get("is_supporting")
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() == "true"
    return bool(v)


def load_musique(
    n_questions: int = 100,
    seed: int = 0,
    cache_dir: str | None = None,
    paragraph_identity: str = "title",
    audit: dict | None = None,
):
    """Return (examples, paragraphs) just like load_hotpotqa.

    `paragraph_identity`: "title" (published protocol — first paragraph
    per title is kept; distinct passages sharing a title COLLAPSE) or
    "title+text" (stable id = title + short content hash; no collapse,
    so recall is exact supporting-paragraph recall). Pass `audit={}` to
    receive collision statistics either way (external review).
    """
    import hashlib
    from datasets import load_dataset

    if paragraph_identity not in ("title", "title+text"):
        raise ValueError("paragraph_identity must be 'title' or 'title+text'")
    ds = load_dataset("dgslibisey/MuSiQue", split="validation",
                      cache_dir=cache_dir)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[:n_questions]

    examples: list[MuSiQueExample] = []
    pool: dict[str, str] = {}
    texts_per_title: dict[str, set[str]] = {}
    gold_collapsed = 0

    def ident(t: str, text: str) -> str:
        if paragraph_identity == "title":
            return t
        return f"{t}::{hashlib.blake2b(text.encode(), digest_size=4).hexdigest()}"

    for i in idx:
        row = ds[int(i)]
        cand_titles, cand_pars, gold_titles = [], [], []
        for p in row["paragraphs"]:
            t = p["title"]
            text = p["paragraph_text"]
            texts_per_title.setdefault(t, set()).add(text)
            key = ident(t, text)
            cand_titles.append(key)
            cand_pars.append((key, text))
            if key not in pool:
                pool[key] = text
            elif pool[key] != text and _is_supporting(p):
                gold_collapsed += 1     # gold passage hidden behind a title
            if _is_supporting(p):
                gold_titles.append(key)

        # n_hops from decomposition length
        decomp = row.get("question_decomposition") or []
        n_hops = len(decomp) if decomp else 2

        examples.append(MuSiQueExample(
            qid=row["id"],
            question=row["question"],
            answer=str(row.get("answer") or ""),
            n_hops=n_hops,
            gold_titles=list(dict.fromkeys(gold_titles)),
            candidate_titles=cand_titles,
            candidate_paragraphs=cand_pars,
        ))
    if audit is not None:
        colliding = {t: len(s) for t, s in texts_per_title.items() if len(s) > 1}
        audit.update({
            "paragraph_identity": paragraph_identity,
            "titles": len(texts_per_title),
            "titles_with_multiple_texts": len(colliding),
            "extra_texts_collapsed": sum(n - 1 for n in colliding.values()),
            "gold_passages_collapsed": gold_collapsed,
        })
    return examples, pool


def retrievable_gold(ex: MuSiQueExample, pool_titles: set[str]) -> list[str]:
    return [t for t in ex.gold_titles if t in pool_titles]
