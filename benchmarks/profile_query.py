"""Profile a single hubmesh query to find the latency bottleneck."""
import sys
import time
from pathlib import Path
import cProfile, pstats
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from hubmesh import Planner, Document
from hubmesh.adapters import InMemoryStore
from hubmesh.kg import build_entity_kg
from hotpotqa_loader import load_hotpotqa
from run_hotpotqa import embed_texts


def main():
    print("Loading HotpotQA n=100 + embedding + building KG (one-time setup)...")
    examples, pool = load_hotpotqa(n_questions=100, seed=0)
    pool_titles = list(pool.keys())
    pool_texts = [pool[t] for t in pool_titles]
    para_vecs = embed_texts(pool_texts)
    query_vecs = embed_texts([ex.question for ex in examples])

    docs = [
        Document(id=t, text=pool[t], vector=para_vecs[i],
                 metadata={"title": t})
        for i, t in enumerate(pool_titles)
    ]
    store = InMemoryStore(docs, k=8)
    import spacy
    nlp = spacy.load("en_core_web_sm")
    kg = build_entity_kg(docs, nlp=nlp)
    planner = Planner(store=store, kg=kg, nlp=nlp)

    # Warm-up
    _ = planner.retrieve(query=examples[0].question,
                         query_vec=query_vecs[0].astype(np.float32),
                         top_k=10, budget_tokens=4000)

    # Coarse timing
    print("\nCoarse timing of 20 queries:")
    times = []
    for i in range(20):
        t0 = time.perf_counter()
        _ = planner.retrieve(query=examples[i].question,
                             query_vec=query_vecs[i].astype(np.float32),
                             top_k=10, budget_tokens=4000)
        times.append(time.perf_counter() - t0)
    print(f"  mean = {np.mean(times)*1000:.1f}ms  "
          f"median = {np.median(times)*1000:.1f}ms  "
          f"p95 = {np.percentile(times, 95)*1000:.1f}ms")

    # cProfile a few queries
    print("\nTop functions by cumulative time (cProfile, 5 queries):")
    profiler = cProfile.Profile()
    profiler.enable()
    for i in range(5):
        _ = planner.retrieve(query=examples[i].question,
                             query_vec=query_vecs[i].astype(np.float32),
                             top_k=10, budget_tokens=4000)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(20)


if __name__ == "__main__":
    main()
