"""Batch-ingest the markdown corpus into a running LightRAG Server.

Per the maintainers (HKUDS/LightRAG#3571): POST /documents/texts with
ids= (RAW title, exact) and file_paths= (sanitized filename) — the server
pipeline gives resumable ingestion (/documents/scan, /documents/reprocess_failed),
which the SDK path lacks.

A local ledger (JSONL of successfully accepted batches) makes THIS script
resumable too: re-running skips batches already accepted by the server.

Usage:
    python ingest_server.py --scope pilot --server http://localhost:9621
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent


def load_ledger(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {json.loads(line)["batch"] for line in path.read_text().splitlines() if line.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--server", default="http://localhost:9621")
    ap.add_argument("--api-key", default=None, help="LightRAG server API key if configured")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-retries", type=int, default=5)
    args = ap.parse_args()

    manifest = json.loads((HERE / f"manifest_{args.scope}.json").read_text())
    corpus_dir = HERE / f"corpus_{args.scope}"
    ledger_path = HERE / f"ingest_ledger_{args.scope}.jsonl"
    done = load_ledger(ledger_path)

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    batches = [manifest[i:i + args.batch] for i in range(0, len(manifest), args.batch)]
    print(f"{len(manifest)} documents in {len(batches)} batches "
          f"({len(done)} already accepted, resuming)")

    t_start = time.time()
    for bi, batch in enumerate(batches):
        if bi in done:
            continue
        payload = {
            "texts": [(corpus_dir / d["filename"]).read_text(encoding="utf-8")
                      for d in batch],
            "ids": [d["title"] for d in batch],
            "file_paths": [d["filename"] for d in batch],
        }
        for attempt in range(args.max_retries):
            try:
                r = requests.post(f"{args.server}/documents/texts",
                                  headers=headers, json=payload, timeout=120)
                if r.status_code < 300:
                    break
                print(f"  batch {bi}: HTTP {r.status_code}: {r.text[:200]}")
            except requests.RequestException as e:
                print(f"  batch {bi}: {e}")
            time.sleep(min(2 ** attempt * 5, 60))
        else:
            print(f"FAILED batch {bi} after {args.max_retries} attempts — "
                  f"fix the server, then re-run (ledger will resume here).")
            sys.exit(1)
        with ledger_path.open("a") as f:
            f.write(json.dumps({"batch": bi, "n": len(batch),
                                "t": time.time()}) + "\n")
        if bi % 20 == 0:
            rate = (bi + 1 - len(done)) / max(time.time() - t_start, 1)
            print(f"  batch {bi + 1}/{len(batches)} accepted "
                  f"({rate:.1f} batches/s submit rate)")

    print("All batches accepted by the server.")
    print("NOTE: acceptance != extraction complete. Extraction is async and "
          "LLM-bound — watch the server's pipeline status, and use "
          "/documents/reprocess_failed for any docs that fail mid-run.")


if __name__ == "__main__":
    main()
