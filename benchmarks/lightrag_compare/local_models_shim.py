"""Local model shim for the LightRAG pilot.

lightrag-hku 1.5.6 has no local-HF binding, so this serves the EXACT
maintainer-specified weights over the wire formats the server does support:

  POST /v1/embeddings   OpenAI format  -> BAAI/bge-m3 (sentence-transformers)
  POST /rerank          Jina format    -> BAAI/bge-reranker-v2-m3 (CrossEncoder)

Same HF cache, same weights, same MPS-safe settings (batch 8, seq cap 1024)
proven in the zero-token grid runs. A lock serializes MPS access.

    python local_models_shim.py   # port 9700
"""
from __future__ import annotations
import threading

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
BATCH = 8
MAX_SEQ = 1024

app = FastAPI()
_lock = threading.Lock()
_models: dict = {}


def get_embedder():
    if "embed" not in _models:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(EMBED_MODEL)
        m.max_seq_length = MAX_SEQ
        _models["embed"] = m
    return _models["embed"]


def get_reranker():
    if "rerank" not in _models:
        from sentence_transformers import CrossEncoder
        _models["rerank"] = CrossEncoder(RERANK_MODEL, max_length=MAX_SEQ)
    return _models["rerank"]


class EmbedReq(BaseModel):
    model: str = EMBED_MODEL
    input: str | list[str]
    encoding_format: str | None = None


class RerankReq(BaseModel):
    model: str = RERANK_MODEL
    query: str
    documents: list[str]
    top_n: int | None = None
    return_documents: bool | None = None


@app.get("/health")
def health():
    return {"status": "ok", "embed": EMBED_MODEL, "rerank": RERANK_MODEL}


@app.post("/v1/embeddings")
def embeddings(req: EmbedReq):
    texts = [req.input] if isinstance(req.input, str) else req.input
    with _lock:
        vecs = get_embedder().encode(
            texts, batch_size=BATCH, normalize_embeddings=True,
            convert_to_numpy=True).astype(np.float32)
    return {
        "object": "list",
        "model": req.model,
        "data": [{"object": "embedding", "index": i, "embedding": v.tolist()}
                 for i, v in enumerate(vecs)],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


@app.post("/rerank")
def rerank(req: RerankReq):
    with _lock:
        scores = get_reranker().predict(
            [(req.query, d) for d in req.documents], batch_size=BATCH)
    scores = 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype=np.float64)))
    order = np.argsort(-scores)
    if req.top_n:
        order = order[:req.top_n]
    results = []
    for i in order:
        r = {"index": int(i), "relevance_score": float(scores[i])}
        if req.return_documents:
            r["document"] = {"text": req.documents[int(i)]}
        results.append(r)
    return {"model": req.model, "results": results,
            "usage": {"total_tokens": 0}}


if __name__ == "__main__":
    import uvicorn
    # preload so the first real request isn't a cold start
    get_embedder()
    get_reranker()
    uvicorn.run(app, host="127.0.0.1", port=9700, log_level="warning")
