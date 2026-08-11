# LightRAG comparison lane — harness

Implements exactly the protocol locked with the LightRAG maintainers in
[HKUDS/LightRAG#3571](https://github.com/HKUDS/LightRAG/issues/3571)
(comments 1–4): Server-API ingestion, markdown `# Title` documents,
`CHUNK_SIZE=1200` + `paragraph_semantic` routing, `mode=mix`,
`chunk_top_k=10`, retrieval measured pre-generation via `/query/data`,
exact doc-id attribution, title+body representation for **all** lanes.

## Pilot runbook (N=500, seed 0)

```bash
# 0. Generate corpus + manifest + questions (no LLM, no server needed)
python gen_md_corpus.py --scope pilot

# 1. Our columns on the title+body representation (zero tokens)
python run_hubmesh_titlebody.py --scope pilot                          # MiniLM
python run_hubmesh_titlebody.py --scope pilot --embed-model BAAI/bge-m3  # naive-on-bge-m3 baseline

# 2. Stand up LightRAG Server with a FRESH working dir + env.pilot as .env
#    (fill in LLM_BINDING_API_KEY first)
#    pip install "lightrag-hku[api]" && lightrag-server

# 3. Ingest (resumable; ledger skips accepted batches on re-run)
python ingest_server.py --scope pilot --server http://localhost:9621
#    ...wait for the server pipeline to finish extraction;
#    use /documents/reprocess_failed for any failures.

# 4. Queries: one main pass + cold-variance sample (100 q x 3 passes).
#    --clear-cache-cmd: maintainer pointer pending; must clear the LLM
#    keyword cache between cold passes or those passes measure the cache.
python run_queries.py --scope pilot --cold-sample 100 --passes 3 \
    --clear-cache-cmd '<pending maintainer pointer>'

# 5. Score offline (re-runnable without re-querying)
python score_eval.py --scope pilot
```

Post to #3571 for maintainer review before anything larger: the recall table,
`env.pilot` (key redacted), and `raw_lightrag_pilot*.jsonl`.

Full run: same steps with `--scope full` (66,581 docs — extraction is the
big token spend; only after pilot review).

## Files

| file | role |
|---|---|
| `gen_md_corpus.py` | pooled paragraphs → `# Title\n\nbody` .md corpus + manifest (raw-title ids) + questions |
| `ingest_server.py` | batch POST `/documents/texts` (`ids`=raw title, `file_paths`=sanitized), ledger-resumable |
| `run_queries.py` | `/query/data` main pass + cache-cold variance sample; dumps RAW JSONL only |
| `score_eval.py` | offline scoring: exact-id recall@k + cross-pass bit-identity check |
| `run_hubmesh_titlebody.py` | hubmesh + naive columns re-measured on title+body (zero tokens) |
| `env.pilot` | locked server config; one REPLACE_ME (LLM key) |
