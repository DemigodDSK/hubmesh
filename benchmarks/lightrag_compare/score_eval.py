"""Score raw /query/data responses offline: supporting-fact recall@k by
exact doc-id, identical to the hubmesh/naive scoring rule.

Chunk -> paragraph attribution: each returned chunk is mapped to its parent
document via (in order) an id field matching a RAW title, or its file_path
matching a manifest filename. With one paragraph per document this mapping
is 1:1 except the few oversize documents that split — a split chunk still
maps to exactly one title, so scoring is unaffected (the realized split
count is reported by the server-side pipeline, not here).

If the response schema differs from expectations, this script fails loudly
and prints the keys it saw — fix the extractor and re-score; the raw JSONL
never has to be re-collected.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Fields that may carry the doc id / file path on a chunk, tried in order.
ID_FIELDS = ("full_doc_id", "doc_id", "id", "source_id")
PATH_FIELDS = ("file_path", "filepath", "source", "filename")


def extract_titles(response: dict, by_title: set[str],
                   by_filename: dict[str, str]) -> list[str]:
    """Ordered list of parent titles for the returned chunks."""
    data = response.get("data", response)
    chunks = data.get("chunks")
    if chunks is None:
        raise KeyError(f"no 'chunks' in response; top-level keys: "
                       f"{sorted(data.keys())[:20]}")
    titles = []
    for ch in chunks:
        t = None
        for fld in ID_FIELDS:
            v = ch.get(fld)
            if isinstance(v, str) and v in by_title:
                t = v
                break
        if t is None:
            for fld in PATH_FIELDS:
                v = ch.get(fld)
                if isinstance(v, str) and v in by_filename:
                    t = by_filename[v]
                    break
        if t is None:
            raise KeyError(f"cannot attribute chunk; chunk keys: "
                           f"{sorted(ch.keys())}; sample values: "
                           f"{ {k: str(ch.get(k))[:60] for k in list(ch)[:6]} }")
        if t not in titles:            # dedupe split-doc siblings, keep order
            titles.append(t)
    return titles


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    top = set(retrieved[:k])
    return sum(1 for g in gold if g in top) / len(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--raw", default=None, help="override raw JSONL path")
    ap.add_argument("--ks", default="2,5,10")
    args = ap.parse_args()
    ks = [int(k) for k in args.ks.split(",")]

    manifest = json.loads((HERE / f"manifest_{args.scope}.json").read_text())
    questions = {q["qid"]: q for q in
                 json.loads((HERE / f"questions_{args.scope}.json").read_text())}
    by_title = {d["title"] for d in manifest}
    by_filename = {d["filename"]: d["title"] for d in manifest}

    raw_path = Path(args.raw) if args.raw else HERE / f"raw_lightrag_{args.scope}.jsonl"
    rows = [json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()]

    per_pass: dict[str, dict[int, list[float]]] = defaultdict(lambda: {k: [] for k in ks})
    per_pass_titles: dict[str, dict[str, list[str]]] = defaultdict(dict)
    n_err = 0
    for row in rows:
        if row.get("error"):
            n_err += 1
            continue
        q = questions[row["qid"]]
        titles = extract_titles(row["response"], by_title, by_filename)
        per_pass_titles[row["pass"]][row["qid"]] = titles
        for k in ks:
            per_pass[row["pass"]][k].append(recall_at_k(titles, q["gold_titles"], k))

    print(f"scored {sum(len(v[ks[0]]) for v in per_pass.values())} rows "
          f"({n_err} errored rows skipped)")
    for tag, res in sorted(per_pass.items()):
        line = "  ".join(f"recall@{k}={np.mean(res[k]):.4f}" for k in ks)
        print(f"[{tag}] n={len(res[ks[0]])}  {line}")

    # Cold-variance: exact result-set agreement across passes on shared qids
    tags = sorted(per_pass_titles)
    cold = [t for t in tags if t.startswith("cold") or t == "main"]
    if len(cold) > 1:
        shared = set.intersection(*(set(per_pass_titles[t]) for t in cold))
        identical = sum(
            1 for qid in shared
            if len({tuple(per_pass_titles[t][qid][:10]) for t in cold}) == 1)
        print(f"[variance] {len(shared)} shared queries across {cold}: "
              f"{identical} bit-identical top-10 lists "
              f"({identical / max(len(shared), 1):.1%})")


if __name__ == "__main__":
    main()
