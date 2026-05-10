"""HotpotQA distractor-split loader for hubmesh benchmarks.

For each question, the distractor split provides 10 candidate paragraphs
(typically 2 gold + 8 distractors) and a list of supporting facts that
identify which (title, sentence_idx) pairs are required to answer.

For retrieval benchmarking we operate at *paragraph* granularity:

  • Each Document is one paragraph: {id=<title>, text=<joined sentences>}
  • The corpus is pooled across all questions in the subset, deduped by title
  • For each question, gold = {titles in supporting_facts that exist in pool}
  • Recall@k = |top_k_titles ∩ gold| / |gold|   (skip Q if gold is empty)

This is a faithful evaluation of *paragraph retrieval* on HotpotQA — the
substrate every multi-hop QA system has to get right before answer
generation can work.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np


@dataclass
class HotpotExample:
    qid: str
    question: str
    answer: str
    qtype: str        # "bridge" | "comparison"
    level: str        # "easy" | "medium" | "hard"
    gold_titles: list[str]                          # supporting_fact titles
    candidate_titles: list[str]                     # the 10 distractor titles
    candidate_paragraphs: list[tuple[str, str]]     # (title, joined_text)


def load_hotpotqa(
    n_questions: int = 100,
    seed: int = 0,
    cache_dir: str | None = None,
):
    """Return (examples, paragraphs) where:
       examples is a list of HotpotExample
       paragraphs is a deduped {title -> joined_text} dict pooled across all
                  examples — the retrieval corpus.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "hotpot_qa", "distractor", split="validation",
        cache_dir=cache_dir,
    )
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[:n_questions]

    examples: list[HotpotExample] = []
    pool: dict[str, str] = {}

    for i in idx:
        row = ds[int(i)]
        # context: {"title": [t1, t2, ...], "sentences": [[s1, s2, ...], ...]}
        titles = row["context"]["title"]
        sentence_lists = row["context"]["sentences"]
        cand_titles = list(titles)
        cand_paragraphs = []
        for t, sents in zip(titles, sentence_lists):
            joined = " ".join(sents).strip()
            cand_paragraphs.append((t, joined))
            if t not in pool:
                pool[t] = joined

        gold_titles = list(dict.fromkeys(row["supporting_facts"]["title"]))

        examples.append(HotpotExample(
            qid=row["id"],
            question=row["question"],
            answer=row["answer"],
            qtype=row["type"],
            level=row["level"],
            gold_titles=gold_titles,
            candidate_titles=cand_titles,
            candidate_paragraphs=cand_paragraphs,
        ))

    return examples, pool


def retrievable_gold(ex: HotpotExample, pool_titles: set[str]) -> list[str]:
    """Subset of gold titles that actually exist in the retrieval pool."""
    return [t for t in ex.gold_titles if t in pool_titles]
