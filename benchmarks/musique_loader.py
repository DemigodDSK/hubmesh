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
):
    """Return (examples, paragraphs) just like load_hotpotqa."""
    from datasets import load_dataset

    ds = load_dataset("dgslibisey/MuSiQue", split="validation",
                      cache_dir=cache_dir)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[:n_questions]

    examples: list[MuSiQueExample] = []
    pool: dict[str, str] = {}

    for i in idx:
        row = ds[int(i)]
        cand_titles, cand_pars, gold_titles = [], [], []
        for p in row["paragraphs"]:
            t = p["title"]
            text = p["paragraph_text"]
            cand_titles.append(t)
            cand_pars.append((t, text))
            if t not in pool:
                pool[t] = text
            if _is_supporting(p):
                gold_titles.append(t)

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
    return examples, pool


def retrievable_gold(ex: MuSiQueExample, pool_titles: set[str]) -> list[str]:
    return [t for t in ex.gold_titles if t in pool_titles]
