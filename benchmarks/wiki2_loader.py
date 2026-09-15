"""2WikiMultihopQA dev loader (held-out evaluation set for the
complementarity protocol; see PROTOCOL_complementarity.md).

Source: the corrected official release `data_ids_april7.zip`
(Alab-NII/2wikimultihop). Downloaded on first use into
`benchmarks/data/2wiki/` (git-ignored); only `dev.json` is extracted.

Each example has 10 context paragraphs (2 or 4 gold). We mirror the
HotpotQA/MuSiQue loaders: paragraphs are pooled across the sampled
questions, but identity is EXACT — `"{title}::{sha1(text)[:8]}"` — so
passages sharing a title never collapse and `id.split("::")[0]` is the
title for the title+body representation. Gold = the distinct ids of the
supporting-fact paragraphs. Question `type` (comparison, inference,
compositional, bridge_comparison) is kept for the protocol's
comparison-type slice.
"""
from __future__ import annotations
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np

DATA_URL = "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1"
DATA_DIR = Path(__file__).resolve().parent / "data" / "2wiki"
DEV_PATH = DATA_DIR / "dev.json"


@dataclass
class Wiki2Example:
    qid: str
    question: str
    answer: str
    qtype: str                                   # comparison | inference | compositional | bridge_comparison
    gold_ids: list[str]                          # distinct supporting-fact paragraph ids
    candidate_ids: list[str]                     # the question's 10 context paragraphs
    candidate_paragraphs: list[tuple[str, str, str]]   # (id, title, joined_text)


def _para_id(title: str, text: str) -> str:
    return f"{title}::{hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]}"


def ensure_dev_file(path: Path = DEV_PATH) -> Path:
    """Download the official archive and extract dev.json if absent."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(DATA_URL, headers={"User-Agent": "hubmesh-benchmarks"})
    with urlopen(req) as r:          # ~260 MB; streamed into memory, then only dev.json kept
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        members = [m for m in z.namelist() if m.endswith("dev.json")]
        if not members:
            raise FileNotFoundError(f"dev.json not found in archive; members: {z.namelist()[:10]}")
        path.write_bytes(z.read(members[0]))
    return path


def load_wiki2(n_questions: int = 1000, seed: int = 0,
               dev_path: Path | str | None = None):
    """Return (examples, pool, titles): pool maps paragraph id -> text,
    titles maps paragraph id -> title."""
    p = ensure_dev_file(Path(dev_path) if dev_path else DEV_PATH)
    rows = json.loads(Path(p).read_text())
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))[:n_questions]

    examples: list[Wiki2Example] = []
    pool: dict[str, str] = {}
    titles: dict[str, str] = {}
    for i in idx:
        row = rows[int(i)]
        paras = []
        for title, sents in row["context"]:
            text = " ".join(s.strip() for s in sents).strip()
            pid = _para_id(title, text)
            paras.append((pid, title, text))
            pool.setdefault(pid, text)
            titles.setdefault(pid, title)
        by_title = {}
        for pid, title, _ in paras:
            by_title.setdefault(title, pid)     # first paragraph carrying the title
        gold = []
        for title, _sent in row["supporting_facts"]:
            pid = by_title.get(title)
            if pid is not None and pid not in gold:
                gold.append(pid)
        examples.append(Wiki2Example(
            qid=str(row["_id"]), question=row["question"],
            answer=row.get("answer", ""),
            qtype=str(row.get("type", "")).lower().replace("-", "_"),
            gold_ids=gold, candidate_ids=[pid for pid, _, _ in paras],
            candidate_paragraphs=paras))
    return examples, pool, titles


def retrievable_gold(ex: Wiki2Example, pool_ids: set[str]) -> list[str]:
    return [g for g in ex.gold_ids if g in pool_ids]
