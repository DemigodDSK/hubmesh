"""Run the evaluation queries against LightRAG Server's /query/data endpoint
and dump RAW responses to JSONL. No scoring here — score_eval.py does that
offline, so a parsing surprise never costs a re-run of paid queries.

Protocol locked in HKUDS/LightRAG#3571:
  mode=mix, chunk_top_k=10, retrieval measured pre-generation via /query/data.

Cold-variance sample (keyword-extraction stochasticity, NOT cache behavior):
  --cold-sample 100 --passes 3 --clear-cache-cmd '<command>'
runs the first 100 questions 3 times, invoking the cache-clear command
between passes. Pass 1 of the main run doubles as cold pass 1.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent


def query_one(server: str, headers: dict, question: str, mode: str,
              chunk_top_k: int, timeout: int = 180) -> dict:
    body = {"query": question, "mode": mode, "chunk_top_k": chunk_top_k}
    r = requests.post(f"{server}/query/data", headers=headers, json=body,
                      timeout=timeout)
    r.raise_for_status()
    return r.json()


def run_pass(server, headers, questions, mode, chunk_top_k, out_path, tag):
    n_err = 0
    with out_path.open("a") as f:
        for i, q in enumerate(questions):
            t0 = time.perf_counter()
            try:
                resp = query_one(server, headers, q["question"], mode, chunk_top_k)
                err = None
            except Exception as e:  # keep going; scorer skips errored rows
                resp, err = None, str(e)
                n_err += 1
            f.write(json.dumps({
                "qid": q["qid"], "pass": tag,
                "latency_s": round(time.perf_counter() - t0, 4),
                "error": err, "response": resp,
            }, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  [{tag}] {i + 1}/{len(questions)} ({n_err} errors)")
    print(f"  [{tag}] done: {len(questions)} queries, {n_err} errors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--server", default="http://localhost:9621")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--mode", default="mix")
    ap.add_argument("--chunk-top-k", type=int, default=10)
    ap.add_argument("--cold-sample", type=int, default=0,
                    help="first N questions get extra cold passes")
    ap.add_argument("--passes", type=int, default=1,
                    help="total passes over the cold sample")
    ap.add_argument("--clear-cache-cmd", default="",
                    help="shell command run between cold passes to clear the "
                         "server's LLM keyword cache (per maintainer pointer)")
    args = ap.parse_args()

    questions = json.loads((HERE / f"questions_{args.scope}.json").read_text())
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    out_path = HERE / f"raw_lightrag_{args.scope}.jsonl"
    print(f"Main pass: {len(questions)} questions -> {out_path}")
    run_pass(args.server, headers, questions, args.mode, args.chunk_top_k,
             out_path, tag="main")

    if args.cold_sample and args.passes > 1:
        sample = questions[:args.cold_sample]
        cold_path = HERE / f"raw_lightrag_{args.scope}_cold.jsonl"
        print(f"Cold-variance sample: {len(sample)} questions x "
              f"{args.passes - 1} extra passes -> {cold_path}")
        for p in range(2, args.passes + 1):
            if args.clear_cache_cmd:
                print(f"  clearing LLM cache: {args.clear_cache_cmd}")
                subprocess.run(args.clear_cache_cmd, shell=True, check=True)
            else:
                print("  WARNING: no --clear-cache-cmd given; this pass will "
                      "hit the keyword cache and is NOT a cold measurement.")
            run_pass(args.server, headers, sample, args.mode,
                     args.chunk_top_k, cold_path, tag=f"cold{p}")


if __name__ == "__main__":
    main()
