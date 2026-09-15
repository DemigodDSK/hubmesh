"""Score raw /query/data responses offline: supporting-fact recall@k by
exact doc-id, identical to the hubmesh/naive scoring rule.

Failure-aware (external review): errored queries are never silently
dropped from averages, and neither are queries that were never written
at all (an interrupted run). Every pass is scored against its INTENDED
roster — the questions file for the main pass, `roster_<scope>_cold.json`
(written by run_queries.py) for cold passes — and reports
  * completion rate (n_completed / n_roster),
  * n_failed (logged errors) and n_missing (in the roster, never written),
  * recall on completed queries only ("completed_only"), and
  * recall counting each failed OR missing query as 0 ("strict"),
and passes are additionally scored on the INTERSECTION of query ids that
completed in every pass ("shared"), so arms are compared on identical
query sets. Appended reruns are deduplicated per (pass, qid), keeping
the LAST occurrence, so a query is never counted twice. Records for ids
outside the question set are reported as "unexpected" and not scored.

Chunk -> paragraph attribution: each returned chunk is mapped to its parent
document via (in order) an id field matching a RAW title, or its file_path
matching a manifest filename. With one paragraph per document this mapping
is 1:1 except the few oversize documents that split — a split chunk still
maps to exactly one title, so scoring is unaffected.

If the response schema differs from expectations, this script fails loudly
and prints the keys it saw — fix the extractor and re-score; the raw JSONL
never has to be re-collected. Writes scored_<scope>.json with a
reproducibility manifest.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from manifest import build_manifest  # noqa: E402

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
    ap.add_argument("--cold-sample", type=int, default=0,
                    help="reconstruct the cold-pass roster as the first N "
                         "questions (run_queries' rule) when "
                         "roster_<scope>_cold.json is absent")
    args = ap.parse_args()
    ks = [int(k) for k in args.ks.split(",")]

    manifest = json.loads((HERE / f"manifest_{args.scope}.json").read_text())
    questions_list = json.loads(
        (HERE / f"questions_{args.scope}.json").read_text())
    questions = {q["qid"]: q for q in questions_list}
    q_order = [q["qid"] for q in questions_list]        # file order
    by_title = {d["title"] for d in manifest}
    by_filename = {d["filename"]: d["title"] for d in manifest}

    raw_path = Path(args.raw) if args.raw else HERE / f"raw_lightrag_{args.scope}.jsonl"
    rows = [json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()]

    # dedupe: last occurrence per (pass, qid) wins (appended reruns)
    latest: dict[tuple[str, str], dict] = {}
    for row in rows:
        latest[(row["pass"], row["qid"])] = row
    n_duplicates = len(rows) - len(latest)

    per_pass_rec: dict[str, dict[str, dict[int, float]]] = defaultdict(dict)
    per_pass_err: dict[str, set[str]] = defaultdict(set)
    per_pass_unexpected: dict[str, set[str]] = defaultdict(set)
    per_pass_titles: dict[str, dict[str, list[str]]] = defaultdict(dict)
    per_pass_latency: dict[str, list[float]] = defaultdict(list)
    for (tag, qid), row in latest.items():
        q = questions.get(qid)
        if q is None:
            per_pass_unexpected[tag].add(qid)   # no gold: cannot be scored
            continue
        if row.get("error"):
            per_pass_err[tag].add(qid)
            continue
        titles = extract_titles(row["response"], by_title, by_filename)
        per_pass_titles[tag][qid] = titles
        per_pass_rec[tag][qid] = {k: recall_at_k(titles, q["gold_titles"], k)
                                  for k in ks}
        per_pass_latency[tag].append(float(row.get("latency_s", 0.0)))

    tags = sorted(set(per_pass_rec) | set(per_pass_err) | set(per_pass_unexpected))

    def roster_for(tag: str) -> tuple[list[str], str]:
        """The query set a pass was SUPPOSED to cover, and where that
        roster came from."""
        if tag == "main":
            return list(q_order), "questions file"
        cold_file = HERE / f"roster_{args.scope}_cold.json"
        if cold_file.exists():
            return list(json.loads(cold_file.read_text())), cold_file.name
        if args.cold_sample:
            return q_order[:args.cold_sample], f"--cold-sample {args.cold_sample}"
        observed = sorted({q for (t, q) in latest if t != "main" and q in questions})
        return observed, ("observed ids — roster not recorded; completion "
                          "may be overstated")

    rosters = {tag: roster_for(tag) for tag in tags}
    shared_ok = (set.intersection(*(set(per_pass_rec[t]) & set(rosters[t][0])
                                    for t in tags))
                 if tags else set())
    out = {
        "manifest": build_manifest(harness=__file__, scope=args.scope,
                                   raw=str(raw_path), n_raw_rows=len(rows),
                                   n_duplicate_rows_dropped=n_duplicates),
        "shared_completed_qids": len(shared_ok),
        "passes": {},
    }

    def agg(recs: dict[str, dict[int, float]], qids, strict_total=None):
        vals = {k: [recs[q][k] for q in qids if q in recs] for k in ks}
        res = {f"recall@{k}": round(float(np.mean(vals[k])), 4)
               if vals[k] else None for k in ks}
        if strict_total is not None:   # failed/missing queries count as 0
            res = {f"recall@{k}": round(float(np.sum(vals[k])) / strict_total, 4)
                   for k in ks}
        return res

    print(f"scored {len(latest)} unique (pass,qid) rows "
          f"({n_duplicates} duplicate rows dropped)")
    for tag in tags:
        expected = set(rosters[tag][0])
        ok = set(per_pass_rec[tag]) & expected
        err = per_pass_err[tag] & expected
        missing = expected - ok - err
        unexpected = ((set(per_pass_rec[tag]) | per_pass_err[tag]
                       | per_pass_unexpected[tag]) - expected)
        total = len(expected)
        block = {
            "roster_source": rosters[tag][1],
            "n_total": total, "n_completed": len(ok), "n_failed": len(err),
            "n_missing": len(missing), "n_unexpected": len(unexpected),
            "completion_rate": round(len(ok) / total, 4) if total else None,
            "completed_only": agg(per_pass_rec[tag], ok),
            "strict_missing_and_failures_as_zero":
                agg(per_pass_rec[tag], ok, strict_total=total) if total else None,
            "shared_query_set": agg(per_pass_rec[tag], shared_ok),
            "latency_s_mean": round(float(np.mean(per_pass_latency[tag])), 4)
            if per_pass_latency[tag] else None,
            "failed_qids": sorted(err)[:50],
            "missing_qids": sorted(missing)[:50],
            "unexpected_qids": sorted(unexpected)[:50],
        }
        out["passes"][tag] = block
        print(f"[{tag}] completed {len(ok)}/{total} "
              f"(failed {len(err)}, missing {len(missing)}, "
              f"unexpected {len(unexpected)}; roster: {rosters[tag][1]}) "
              f"| completed-only {block['completed_only']} "
              f"| strict {block['strict_missing_and_failures_as_zero']} "
              f"| shared(n={len(shared_ok)}) {block['shared_query_set']}")

    # Cold-variance: exact result-set agreement across passes on shared qids
    cold = [t for t in tags if t.startswith("cold") or t == "main"]
    if len(cold) > 1 and shared_ok:
        identical = sum(
            1 for qid in shared_ok
            if len({tuple(per_pass_titles[t][qid][:10]) for t in cold}) == 1)
        out["variance"] = {"passes": cold, "n_shared": len(shared_ok),
                           "bit_identical_top10": identical,
                           "fraction": round(identical / len(shared_ok), 4)}
        print(f"[variance] {len(shared_ok)} shared queries across {cold}: "
              f"{identical} bit-identical top-10 lists "
              f"({identical / len(shared_ok):.1%})")

    path = HERE / f"scored_{args.scope}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
