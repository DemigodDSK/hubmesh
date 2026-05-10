# Contributing to hubmesh

Thanks for your interest. hubmesh is in pre-alpha — the API will change, and
contributions that help shape it are welcome.

## Development setup

```bash
git clone https://github.com/DemigodDSK/hubmesh.git
cd hubmesh
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,benchmarks]"
python -m spacy download en_core_web_sm   # for entity-linked KG mode
```

Run tests:

```bash
pytest tests/ -v
```

Run the synthetic demo:

```bash
python examples/synthetic_demo.py
```

Run the HotpotQA benchmark (downloads ~50MB on first run, cached after):

```bash
python benchmarks/run_hotpotqa.py --n 100        # kNN-graph mode
python benchmarks/run_hotpotqa.py --n 100 --kg   # entity-linked KG mode
```

## Where help is most welcome

- **Adapter implementations.** The `VectorStore` protocol in
  `src/hubmesh/adapters/base.py` defines what's needed. Pinecone, Qdrant,
  Weaviate, pgvector, Chroma all welcome.
- **Better entity linking.** The current canonicalize-and-substring matcher
  in `src/hubmesh/kg.py` is intentionally crude. Embedding-based linking,
  BLINK-style learned linking, or LLM-extracted triples are all upgrades.
- **Benchmarks.** Run hubmesh on your data and report results; or extend
  `benchmarks/` with MuSiQue, 2WikiMultiHopQA, or a BEIR subset.
- **Latency optimization.** Sparse PPR, query-time spaCy caching,
  candidate prefiltering by cosine before PPR.

## Issue / PR style

Small focused PRs are easier to review. Include a one-line summary of *what*
and *why*. Tests for new behavior. No large dependencies without discussion
— hubmesh is meant to stay light.
