"""Complementarity experiment — the pre-registered protocol in
PROTOCOL_complementarity.md. Read that file first; every knob here is
fixed there (Amendments 1 and 2 included).

Arms: dense / BM25 / hybrid-RRF references; hybrid top-50 -> rerank
(equal-budget control, A4); hybrid top-30 ∪ hubmesh top-20 -> rerank
(treatment, A6); dense-based union (A7); hubmesh top-50 -> rerank (A8,
variable-count: the token-budgeted retrieve call returns fewer than 50
for many queries — recorded per query); hubmesh top-5 (A5). Primary
metric: complete-evidence@5. Primary contrast: A6 − A4 with a paired
bootstrap and the decision rule from the protocol.

Two models cannot be resident together on an 8 GB machine, so the run is
staged: stage R (embedder + graph resident) produces candidates and
per-query retrieval timings; stage K (reranker resident) scores the
candidate lists. `--stage all` runs both in one process, unloading the
embedder before loading the reranker; `--stage retrieve` / `--stage
rerank` allow resuming from the stage file, which is identity-checked
before reuse and never rewritten by stage K.

Stage K (Amendment 2): every pair is tokenized exactly as the library's
CrossEncoder does (same tokenizer, weights, pair truncation, max length
512) but scored through fixed-shape bucketed batches, because the
library's dynamic padding retained Metal driver memory per distinct
shape and ran out after ~25 queries. Ranking uses raw logits (monotone
in the library's sigmoid; avoids fp16 saturation ties). Reranking
latency is measured PER ARM — each reranked arm's own list scored from
scratch and timed — on the first `--rerank-timing-queries` queries; the
scores used for ranking come from one untimed pass over the union of the
arms' candidates, cached per (question, paragraph) so the two embedder
configurations of a dataset share identical pair scores.

Latency is composed PER QUERY from stage times before percentiles are
taken, and is labelled an estimate in the output.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from manifest import build_manifest, _git  # noqa: E402

RRF_K = 60
TOPN_FUSE = 200
BASE_N, HUB_N, RERANK_N = 30, 20, 50
REF_N = 10                # prefix kept for the reference arms (top-5 arms; @10 needs 10)
KS = (5, 10)
ARMS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
ARM_DESC = {
    "A1": "dense top-5 (10-prefix kept for the @10 metrics)",
    "A2": "BM25 top-5 (10-prefix kept for the @10 metrics)",
    "A3": "hybrid RRF top-5 (10-prefix kept for the @10 metrics)",
    "A4": "hybrid top-50 -> rerank -> top-5 (equal-budget control)",
    "A5": "hubmesh top-5 (10-prefix kept for the @10 metrics)",
    "A6": "hybrid top-30 ∪ hubmesh top-20 (backfilled to 50) -> rerank -> top-5 (treatment)",
    "A7": "dense top-30 ∪ hubmesh top-20 (backfilled to 50) -> rerank -> top-5",
    "A8": ("hubmesh top-50 -> rerank -> top-5 — VARIABLE-COUNT, token-budgeted arm: one "
           "retrieve(top_k=50, budget_tokens=10_000) call, whose packing budget binds for "
           "many queries, so it reranks fewer than 50 candidates (count recorded per query)"),
}
RERANKED = {"A4", "A6", "A7", "A8"}
STAGES_USED = {           # which retrieval stage times each arm pays
    "A1": ("embed", "dense"), "A2": ("bm25",), "A3": ("embed", "dense", "bm25", "rrf"),
    "A4": ("embed", "dense", "bm25", "rrf", "rerank"),
    "A5": ("embed", "hubmesh"),
    "A6": ("embed", "dense", "bm25", "rrf", "hubmesh", "rerank"),
    "A7": ("embed", "dense", "hubmesh", "rerank"),
    "A8": ("embed", "hubmesh", "rerank"),
}
RERANK_BUCKETS = (128, 256, 384, 512)   # padded sequence lengths the reranker ever sees
RERANK_MAX_LENGTH = 512
_TOKEN_RE = re.compile(r"\w+")


# ---- pure functions (unit-tested in tests/test_complementarity_protocol.py) --

def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal-rank fusion of ranked id lists; ties broken by id."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return [d for d, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]


def union_backfill(base: list[str], extra: list[str], base_n: int = BASE_N,
                   extra_n: int = HUB_N, total: int = RERANK_N) -> list[str]:
    """base[:base_n], then extra[:extra_n] in extra's order skipping
    duplicates, then base[base_n:] until `total` distinct ids."""
    out: list[str] = []
    seen: set[str] = set()
    for d in list(base[:base_n]) + list(extra[:extra_n]) + list(base[base_n:]):
        if d in seen:
            continue
        out.append(d)
        seen.add(d)
        if len(out) >= total:
            break
    return out


def complete_evidence(retrieved: list[str], gold: list[str], k: int) -> float:
    """1.0 iff every distinct gold id is among the top-k distinct retrieved
    ids. A gold id that exists in no retrievable list (absent from the
    pool) can never be matched, so the question scores 0 — the protocol's
    'absent from the pool counts as failure'."""
    if not gold:
        return 0.0
    top = set(retrieved[:k])
    return 1.0 if all(g in top for g in set(gold)) else 0.0


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(retrieved[:k])
    gold_set = set(gold)
    return sum(1 for g in gold_set if g in top) / len(gold_set)


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 10_000,
                     seed: int = 0) -> dict | None:
    d = (np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) * 100.0
    if len(d) == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return {"mean": round(float(d.mean()), 2),
            "lo": round(float(np.percentile(means, 2.5)), 2),
            "hi": round(float(np.percentile(means, 97.5)), 2), "n": int(len(d))}


def decide(contrast: dict | None, threshold: float = 3.0) -> str:
    """Protocol decision rule on the primary contrast (points)."""
    if contrast is None:
        return "inconclusive"
    if contrast["mean"] >= threshold and contrast["lo"] > 0:
        return "pass"
    if contrast["hi"] < threshold:
        return "no useful gain"
    return "inconclusive"


def compose_latency(stage_times: list[dict], stages: tuple[str, ...]) -> dict:
    """Per-query sums of the named stage times, THEN median / p95."""
    sums = np.array([sum(float(st.get(s, 0.0)) for s in stages) for st in stage_times])
    if len(sums) == 0:
        return {"median_ms": None, "p95_ms": None}
    return {"median_ms": round(float(np.median(sums)) * 1000, 2),
            "p95_ms": round(float(np.percentile(sums, 95)) * 1000, 2)}


def arm_stage_times(rec: dict, arm: str) -> dict | None:
    """The stage-time dict an arm pays for one query. Reranked arms use
    THEIR OWN measured rerank time (`rerank_by_arm`, Amendment 2); queries
    outside the timing subsample have none and are excluded (None)."""
    st = dict(rec["stage_times"])
    if arm in RERANKED:
        by_arm = st.get("rerank_by_arm") or {}
        if arm not in by_arm:
            return None
        st["rerank"] = by_arm[arm]
    return st


def plan_fixed_shape_batches(lengths: list[int], buckets: tuple[int, ...] = RERANK_BUCKETS,
                             batch: int = 8) -> list[tuple[int, list[int], int]]:
    """Fixed-shape batching plan: every pair goes to the smallest bucket
    that holds its token length; each bucket is cut into batches of
    exactly `batch` rows, a short last batch padded by repeating its final
    index. Returns [(bucket, row_indices, n_real), ...]; the first n_real
    indices of every batch are distinct real pairs, and each input index
    appears exactly once as a real row. Lengths above the last bucket are
    an error (the caller truncates to the last bucket first)."""
    if not buckets or any(b <= 0 for b in buckets) or list(buckets) != sorted(set(buckets)):
        raise ValueError("buckets must be increasing positive ints")
    over = [i for i, n in enumerate(lengths) if n > buckets[-1] or n <= 0]
    if over:
        raise ValueError(f"{len(over)} lengths outside (0, {buckets[-1]}]")
    plan = []
    prev = 0
    for b in buckets:
        idx = [i for i, n in enumerate(lengths) if prev < n <= b]
        prev = b
        for s in range(0, len(idx), batch):
            chunk = idx[s:s + batch]
            plan.append((b, chunk + [chunk[-1]] * (batch - len(chunk)), len(chunk)))
    return plan


def validate_stage(stage: dict, expected: dict, qids: list[str], pool_ids) -> dict:
    """Identity checks before a stage file is reused (Amendment 2). Raises
    ValueError with every problem found; returns what was checked."""
    problems: list[str] = []
    checked: dict = {}
    for k in ("dataset", "n", "seed", "embed_model"):
        if k in stage:
            checked[k] = stage[k]
            if stage[k] != expected[k]:
                problems.append(f"{k}: stage file {stage[k]!r} != this run {expected[k]!r}")
        else:
            checked[k] = "absent (stage file predates Amendment 2; identity rests on the qid and pool checks)"
    recs = stage.get("records", [])
    got = [r["qid"] for r in recs]
    if got != list(qids):
        problems.append(f"qid list differs from the loaded sample ({len(got)} vs {len(qids)} or out of order)")
    pool = set(pool_ids)
    bad = sum(1 for r in recs for a in ARMS for d in r["cand"][a] if d not in pool)
    if bad:
        problems.append(f"{bad} candidate ids are not in the pool")
    want = min(RERANK_N, len(pool))
    for a in ("A4", "A6", "A7"):
        short = sum(1 for r in recs if len(r["cand"][a]) != want or len(set(r["cand"][a])) != want)
        if short:
            problems.append(f"{a}: {short} queries without exactly {want} distinct candidates")
    a8 = [len(r["cand"]["A8"]) for r in recs]
    if any(n > RERANK_N for n in a8) or any(len(set(r["cand"]["A8"])) != len(r["cand"]["A8"]) for r in recs):
        problems.append("A8: more than 50 or duplicate candidates")
    for a, s in (("A1", "A7"), ("A3", "A4"), ("A5", "A8")):
        mism = sum(1 for r in recs if r["cand"][s][:len(r["cand"][a])] != r["cand"][a])
        if mism:
            problems.append(f"{a} is not a prefix of {s} in {mism} queries")
    if problems:
        raise ValueError("stage file rejected: " + "; ".join(problems))
    return {"checked": checked, "n_records": len(recs), "qid_order_matches_sample": True,
            "all_candidates_in_pool": True, "A4_A6_A7_exactly_50_distinct": True,
            "A8_count_min": min(a8) if a8 else None,
            "A8_queries_below_50": sum(1 for n in a8 if n < RERANK_N),
            "prefix_relations_hold": "A1<A7, A3<A4, A5<A8"}


def extend_reference_arms(records: list[dict], bm25_top_of: dict[str, list[str]] | None,
                          ref_n: int = REF_N) -> int:
    """Stage files written before Amendment 2 kept only the top-5 of the
    reference arms, so their @10 metrics equalled @5. dense / hybrid /
    hubmesh top-10 are exact prefixes of A7 / A4 / A8 (union_backfill
    copies base[:30] first; A4 = hybrid[:50]; A8 = hub[:50]); BM25 is
    recomputed deterministically. Every stored 5-prefix must match or the
    file is rejected. Returns the number of lists extended."""
    n_ext = 0
    for r in records:
        for a, s in (("A1", "A7"), ("A3", "A4"), ("A5", "A8")):
            old = r["cand"][a]
            if len(old) >= ref_n or len(r["cand"][s]) <= len(old):
                continue
            new = list(r["cand"][s][:ref_n])
            if new[:len(old)] != list(old):
                raise ValueError(f"{r['qid']}: stored {a} is not a prefix of {s}")
            r["cand"][a] = new
            n_ext += 1
        old = r["cand"]["A2"]
        if len(old) < ref_n:
            if bm25_top_of is None or r["qid"] not in bm25_top_of:
                raise ValueError(f"{r['qid']}: A2 has {len(old)} ids and no BM25 ranking was supplied")
            new = list(bm25_top_of[r["qid"]][:ref_n])
            if new[:len(old)] != list(old):
                raise ValueError(f"{r['qid']}: recomputed BM25 ranking does not start with the stored A2")
            r["cand"]["A2"] = new
            n_ext += 1
    return n_ext


def refresh_gold(records: list[dict], gold_of: dict[str, list[str]], pool_ids) -> dict:
    """Replace the stored gold with the FULL gold list so a gold paragraph
    absent from the pool counts as failure (protocol Metrics; Amendment
    2). The stored list must equal either the full list or its
    retrievable subset, otherwise the stage file is rejected."""
    pool = set(pool_ids)
    unresolved = 0
    affected = 0
    for r in records:
        full = list(gold_of[r["qid"]])
        retrievable = [g for g in full if g in pool]
        if list(r["gold"]) not in (retrievable, full):
            raise ValueError(f"{r['qid']}: stored gold {r['gold']} is neither the full gold {full} "
                             f"nor its retrievable subset {retrievable}")
        missing = len(full) - len(retrievable)
        r["gold"], r["n_gold"], r["n_gold_unresolved"] = full, len(full), missing
        unresolved += missing
        affected += bool(missing)
    return {"gold_ids_unresolved": unresolved, "queries_with_unresolved_gold": affected}


def _mps_empty_cache() -> None:
    """Release PyTorch's cached Metal blocks. The caching allocator keeps
    freed blocks across variable-length batches; on an 8 GB machine that
    grew past the allocation ceiling (Amendment 1) even though the fp16
    model itself needs ~1.1 GB."""
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _mps_memory() -> dict:
    """Tensor memory PyTorch holds vs. everything the Metal driver holds for
    the process (caches, graphs). Reported, not interpreted."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return {"allocated_gb": round(torch.mps.current_allocated_memory() / 2**30, 3),
                    "driver_gb": round(torch.mps.driver_allocated_memory() / 2**30, 3)}
    except Exception:
        pass
    return {}


def encode_chunked(model, texts: list[str], batch_size: int, chunk: int = 512) -> np.ndarray:
    """Length-sorted, chunked encoding with a cache release after every
    chunk: bounded accelerator memory, same embeddings, original order."""
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    out = None
    for s in range(0, len(order), chunk):
        idx = order[s:s + chunk]
        embs = model.encode([texts[i] for i in idx], batch_size=batch_size,
                            show_progress_bar=False, normalize_embeddings=True,
                            convert_to_numpy=True).astype(np.float32)
        if out is None:
            out = np.zeros((len(texts), embs.shape[1]), dtype=np.float32)
        out[idx] = embs
        _mps_empty_cache()
        print(f"      embedded {min(s + chunk, len(order))}/{len(order)}", flush=True)
    return out if out is not None else np.zeros((0, 0), dtype=np.float32)


def protocol_sha256(path: Path = HERE / "PROTOCOL_complementarity.md") -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _sha256_of_ids(ids) -> str:
    return hashlib.sha256("\n".join(map(str, ids)).encode("utf-8")).hexdigest()


# ---- reranker: fixed-shape bucketed scoring + score cache -------------------

class FixedShapeReranker:
    """Cross-encoder scoring with a fixed set of padded shapes.

    Uses the CrossEncoder's own tokenizer and weights. Each pair is
    tokenized exactly as the library does it (pair truncation
    'longest_first' to `max_length`), assigned to the smallest bucket that
    holds it, and scored in constant-size batches padded with duplicate
    pairs, so the model only ever sees len(buckets) shapes. Right padding
    with an attention mask; the classification head reads position 0, so
    padding cannot change a real row's logit beyond fp16 batch effects.
    Scores are raw logits (the library's default sigmoid is monotone in
    them, so rankings agree except where fp16 sigmoid saturates to ties).
    """

    def __init__(self, ce, device: str, buckets=RERANK_BUCKETS, batch: int = 8,
                 max_length: int = RERANK_MAX_LENGTH):
        self.ce = ce
        self.model = ce[0].auto_model
        self.tok = ce.tokenizer
        self.device = device
        self.buckets = tuple(buckets)
        self.batch = int(batch)
        self.max_length = int(max_length)
        self.model.train(False)          # inference behaviour for dropout / norm layers
        self.n_pairs = self.n_batches = self.n_padded_rows = 0

    def dtype(self) -> str:
        return str(next(self.model.parameters()).dtype)

    def lengths(self, pairs: list[tuple[str, str]]) -> list[int]:
        enc = self.tok([q for q, _ in pairs], [d for _, d in pairs], truncation="longest_first",
                       max_length=self.max_length, padding=False)
        return [len(x) for x in enc["input_ids"]]

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        import torch
        if not pairs:
            return []
        out: list[float | None] = [None] * len(pairs)
        for bucket, idx, n_real in plan_fixed_shape_batches(self.lengths(pairs), self.buckets, self.batch):
            feats = self.tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx],
                             truncation="longest_first", max_length=bucket, padding="max_length",
                             return_tensors="pt")
            feats = {k: v.to(self.device) for k, v in feats.items()
                     if k in ("input_ids", "attention_mask", "token_type_ids")}
            with torch.inference_mode():
                logits = self.model(**feats).logits
            vals = logits.float().reshape(len(idx), -1)[:, 0].cpu().tolist()
            for i, v in zip(idx[:n_real], vals[:n_real]):
                out[i] = float(v)
            self.n_batches += 1
            self.n_padded_rows += len(idx) - n_real
        self.n_pairs += len(pairs)
        return out  # type: ignore[return-value]

    def warm_up(self) -> dict:
        """One batch per bucket so kernel compilation is not timed."""
        t0 = time.perf_counter()
        pairs = []
        for b in self.buckets:
            words = b
            while True:
                doc = " ".join(["a"] * words)
                if self.lengths([("warm-up question", doc)])[0] <= b:
                    break
                words = max(1, int(words * 0.8))
            pairs.append(("warm-up question", doc))
        compiled = sorted({bucket for bucket, _, _ in plan_fixed_shape_batches(self.lengths(pairs), self.buckets, self.batch)})
        self.score(pairs)
        return {"buckets_compiled": compiled, "seconds": round(time.perf_counter() - t0, 2)}


class ScoreCache:
    """(qid, doc_id) -> logit, persisted with the provenance that makes the
    scores reusable: model, device, dtype, buckets, batch, max length,
    dataset sample. Both embedder configurations of a dataset share it."""

    def __init__(self, path: Path | None, provenance: dict):
        self.path = Path(path) if path else None
        self.provenance = provenance
        self.scores: dict[tuple[str, str], float] = {}
        self.loaded = 0
        self.hits = 0
        if self.path and self.path.exists():
            blob = json.loads(self.path.read_text())
            if blob.get("provenance") == provenance:
                self.scores = {tuple(k.split("\t", 1)): float(v) for k, v in blob["scores"].items()}
                self.loaded = len(self.scores)
            else:
                print(f"      score cache {self.path} has different provenance; not reused", flush=True)

    def get(self, qid: str, doc: str) -> float | None:
        v = self.scores.get((qid, doc))
        if v is not None:
            self.hits += 1
        return v

    def put(self, qid: str, doc: str, v: float) -> None:
        self.scores[(qid, doc)] = float(v)

    def save(self) -> None:
        if not self.path:
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"provenance": self.provenance,
                                   "scores": {f"{q}\t{d}": v for (q, d), v in self.scores.items()}}))
        os.replace(tmp, self.path)


def smoke_parity(ce, reranker: FixedShapeReranker, rec: dict, texts: dict, questions: dict,
                 batch_size: int) -> dict:
    """The library path (dynamic padding, CrossEncoder.predict with the
    identity activation) vs. the fixed-shape path on one query's union of
    candidates: same weights, same tokenizer, only the batching differs."""
    import torch
    q = questions[rec["qid"]]
    docs = list(dict.fromkeys(d for a in sorted(RERANKED) for d in rec["cand"][a]))
    pairs = [(q, texts[d]) for d in docs]
    lib = [float(x) for x in np.asarray(ce.predict(pairs, batch_size=batch_size,
                                                    activation_fn=torch.nn.Identity(),
                                                    show_progress_bar=False)).reshape(-1)]
    fx = reranker.score(pairs)
    diffs = [abs(a - b) for a, b in zip(lib, fx)]
    order_lib = sorted(range(len(docs)), key=lambda i: (-lib[i], i))
    order_fx = sorted(range(len(docs)), key=lambda i: (-fx[i], i))
    return {"qid": rec["qid"], "n_pairs": len(pairs), "max_abs_diff": round(max(diffs), 6),
            "mean_abs_diff": round(float(np.mean(diffs)), 6),
            "same_full_order": order_lib == order_fx, "same_top5": order_lib[:5] == order_fx[:5],
            "same_top10": order_lib[:10] == order_fx[:10],
            "library_path": f"CrossEncoder.predict(activation_fn=Identity, batch_size={batch_size})"}


# ---- data ------------------------------------------------------------------

def load_data(dataset: str, n: int, seed: int):
    """Return (examples, pool_ids, texts, titles, gold_of, qtype_of, hop_of).
    gold_of holds the FULL gold list per question — including any gold
    paragraph absent from the pool, which the metrics count as failure."""
    if dataset == "2wiki":
        from wiki2_loader import load_wiki2
        examples, pool, titles = load_wiki2(n_questions=n, seed=seed)
        ids = list(pool.keys())
        gold_of = {ex.qid: list(ex.gold_ids) for ex in examples}
        qtype_of = {ex.qid: ex.qtype for ex in examples}
        hop_of = {}
    elif dataset == "hotpotqa":
        from hotpotqa_loader import load_hotpotqa
        examples, pool = load_hotpotqa(n_questions=n, seed=seed)
        ids = list(pool.keys())
        titles = {t: t for t in ids}
        gold_of = {ex.qid: list(ex.gold_titles) for ex in examples}
        qtype_of = {ex.qid: ex.qtype for ex in examples}
        hop_of = {}
    elif dataset == "musique":
        from musique_loader import load_musique
        examples, pool = load_musique(n_questions=n, seed=seed,
                                      paragraph_identity="title+text")
        ids = list(pool.keys())
        titles = {t: t.split("::")[0] for t in ids}
        gold_of = {ex.qid: list(ex.gold_titles) for ex in examples}
        qtype_of = {ex.qid: f"{ex.n_hops}-hop" for ex in examples}
        hop_of = {ex.qid: ex.n_hops for ex in examples}
    else:
        raise ValueError(dataset)
    texts = {i: f"{titles[i]}\n\n{pool[i]}" for i in ids}
    return examples, ids, texts, titles, gold_of, qtype_of, hop_of


def bm25_rankings(examples, ids: list[str], texts: dict, topn: int = TOPN_FUSE) -> dict[str, list[str]]:
    """Deterministic BM25 ranking per question (protocol: rank_bm25 defaults,
    lowercase \\w+ tokens) — the same computation stage R performs."""
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([tokenize(texts[i]) for i in ids])
    id_arr = np.array(ids)
    out = {}
    for ex in examples:
        scores = bm25.get_scores(tokenize(ex.question))
        order = np.argsort(-scores, kind="stable")[:topn]
        out[ex.qid] = [str(id_arr[j]) for j in order]
    return out


# ---- stage R: embedder + graph resident ------------------------------------

def stage_retrieve(args, examples, ids, texts, gold_of):
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    from hubmesh import Planner, Document
    from hubmesh.adapters import InMemoryStore
    from hubmesh.kg import build_entity_kg
    import spacy

    load_kwargs = {}
    if args.embed_dtype == "fp16":
        import torch
        # Amendment 1: load the weights AS fp16 — converting after an fp32
        # load leaves a 2 GB fp32 copy on the accelerator and trips the
        # Metal memory ceiling before embedding starts.
        load_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    model = SentenceTransformer(args.embed_model, device=args.embed_device, **load_kwargs)
    if args.embed_max_seq_len:
        model.max_seq_length = args.embed_max_seq_len
    if args.embed_dtype == "fp16":
        model.half()            # no-op for the transformer; covers pooling/dense modules
    t0 = time.perf_counter()
    para_vecs = encode_chunked(model, [texts[i] for i in ids], args.embed_batch_size)
    embed_pool_s = time.perf_counter() - t0

    docs = [Document(id=i, text=texts[i], vector=para_vecs[j]) for j, i in enumerate(ids)]
    store = InMemoryStore(docs, k=8)
    nlp = spacy.load("en_core_web_sm")
    t0 = time.perf_counter()
    kg = build_entity_kg(docs, nlp=nlp)
    kg_build_s = time.perf_counter() - t0
    planner = Planner(store=store, kg=kg, nlp=nlp)
    bm25 = BM25Okapi([tokenize(texts[i]) for i in ids])
    id_arr = np.array(ids)
    pool = set(ids)

    records = []
    for qi, ex in enumerate(examples):
        gold = list(gold_of[ex.qid])
        st: dict[str, float] = {}
        t0 = time.perf_counter()
        qvec = model.encode([ex.question], normalize_embeddings=True,
                            convert_to_numpy=True)[0].astype(np.float32)
        st["embed"] = time.perf_counter() - t0
        if qi % 50 == 0:
            _mps_empty_cache()
            print(f"      query {qi}/{len(examples)}", flush=True)
        t0 = time.perf_counter()
        dense = [d for d, _ in store.search(qvec, top_k=TOPN_FUSE)]
        st["dense"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        scores = bm25.get_scores(tokenize(ex.question))
        order = np.argsort(-scores, kind="stable")[:TOPN_FUSE]
        lex = [str(id_arr[j]) for j in order]
        st["bm25"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        hybrid = rrf_fuse([dense, lex])
        st["rrf"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        res = planner.retrieve(query=ex.question, query_vec=qvec, top_k=RERANK_N,
                               budget_tokens=10_000)
        st["hubmesh"] = time.perf_counter() - t0
        hub = [s.doc.id for s in res.sources]
        n_seeds = len(res.debug.get("ppr_seeds", [])) if res.debug else 0
        records.append({
            "qid": ex.qid, "gold": gold, "n_gold": len(gold),
            "n_gold_unresolved": sum(1 for g in gold if g not in pool),
            "n_seeds": n_seeds,
            "overlap_hybrid30_hub20": len(set(hybrid[:BASE_N]) & set(hub[:HUB_N])),
            "gold_docs_without_entities": sum(
                1 for g in gold if g in pool and not kg.doc_to_entities.get(g)),
            "stage_times": st,
            "cand": {
                "A1": dense[:REF_N], "A2": lex[:REF_N], "A3": hybrid[:REF_N],
                "A4": hybrid[:RERANK_N], "A5": hub[:REF_N],
                "A6": union_backfill(hybrid, hub),
                "A7": union_backfill(dense, hub),
                "A8": hub[:RERANK_N],
            },
        })
    stage = {
        "dataset": args.dataset, "n": args.n, "seed": args.seed,
        "qids_sha256": _sha256_of_ids(ex.qid for ex in examples),
        "pool_sha256": _sha256_of_ids(ids),
        "protocol_sha256": protocol_sha256(), "git_commit": _git("rev-parse", "HEAD"),
        "embed_model": args.embed_model, "embed_device": args.embed_device or "auto",
        "embed_dtype": args.embed_dtype,
        "embed_batch_size": args.embed_batch_size,
        "embed_max_seq_len": args.embed_max_seq_len,
        "pool": len(ids), "embed_pool_s": round(embed_pool_s, 1),
        "kg_nodes": kg.graph.number_of_nodes(), "kg_edges": kg.graph.number_of_edges(),
        "kg_build_s": round(kg_build_s, 1), "records": records,
    }
    del model, planner, store, kg, bm25
    gc.collect()
    _mps_empty_cache()
    return stage


# ---- stage K: reranker resident --------------------------------------------

def stage_rerank(args, stage, texts, questions):
    import torch
    from sentence_transformers import CrossEncoder
    ce_kwargs = {}
    if args.rerank_dtype == "fp16":
        ce_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}   # load AS fp16 (Amendment 1)
    ce = CrossEncoder(args.rerank_model, max_length=RERANK_MAX_LENGTH,
                      device=args.rerank_device, **ce_kwargs)
    device = args.rerank_device or str(ce[0].auto_model.device)
    reranker = FixedShapeReranker(ce, device=device, buckets=args.rerank_buckets,
                                  batch=args.rerank_batch_size, max_length=RERANK_MAX_LENGTH)
    dtype_actual = reranker.dtype()
    print(f"      reranker on {device}, parameters {dtype_actual}, buckets {reranker.buckets}, "
          f"batch {reranker.batch}, {_mps_memory()}", flush=True)
    provenance = {"rerank_model": args.rerank_model, "device": device, "dtype": dtype_actual,
                  "buckets": list(reranker.buckets), "batch": reranker.batch,
                  "max_length": RERANK_MAX_LENGTH, "scores": "raw logits",
                  "dataset": stage.get("dataset", args.dataset), "n": stage.get("n", args.n),
                  "seed": stage.get("seed", args.seed), "representation": "title+body"}
    cache = ScoreCache(args.rerank_cache, provenance)
    if cache.loaded:
        print(f"      score cache: {cache.loaded} pair scores reused from {args.rerank_cache}", flush=True)
    recs = stage["records"]

    warm = reranker.warm_up()
    parity = smoke_parity(ce, reranker, recs[0], texts, questions, args.rerank_batch_size)
    print(f"      smoke parity (library dynamic-padding path vs fixed-shape, same weights): {parity}", flush=True)
    if parity["max_abs_diff"] > args.rerank_parity_tolerance:
        raise RuntimeError(f"fixed-shape scores differ from the library path by {parity['max_abs_diff']} "
                           f"(> {args.rerank_parity_tolerance}); refusing to run")
    _mps_empty_cache()

    # ---- timing pass: each reranked arm's own list, scored from scratch and
    # timed, on the first T queries (Amendment 2). Scores are kept only to
    # measure how much batch composition moves an fp16 logit.
    n_timing = min(args.rerank_timing_queries, len(recs))
    timing_scores: dict[tuple[str, str], float] = {}
    mem_samples = [{"at": "start", **_mps_memory()}]
    t_pass = time.perf_counter()
    n0 = reranker.n_pairs
    for qi in range(n_timing):
        rec = recs[qi]
        q = questions[rec["qid"]]
        by_arm = {}
        for a in sorted(RERANKED):
            pairs = [(q, texts[d]) for d in rec["cand"][a]]
            t0 = time.perf_counter()
            s = reranker.score(pairs)
            by_arm[a] = time.perf_counter() - t0
            for d, v in zip(rec["cand"][a], s):
                timing_scores[(rec["qid"], d)] = v
        rec["stage_times"]["rerank_by_arm"] = by_arm
        _mps_empty_cache()
        if qi % 10 == 0 or qi == n_timing - 1:
            mem_samples.append({"at": f"timing q{qi}", **_mps_memory()})
            rate = (reranker.n_pairs - n0) / max(1e-9, time.perf_counter() - t_pass)
            print(f"      timing {qi + 1}/{n_timing}: "
                  + " ".join(f"{a}={by_arm[a] * 1000:.0f}ms" for a in sorted(by_arm))
                  + f" | {rate:.2f} pairs/s | {_mps_memory()}", flush=True)
    timing_pass_s = time.perf_counter() - t_pass

    # ---- scoring pass: the union of the reranked arms' candidates, groups of
    # queries so batches are full; cached per (question, paragraph).
    t_pass = time.perf_counter()
    n0 = reranker.n_pairs
    group = 4
    for g in range(0, len(recs), group):
        todo: list[tuple[str, str]] = []
        for rec in recs[g:g + group]:
            for d in dict.fromkeys(d for a in sorted(RERANKED) for d in rec["cand"][a]):
                if cache.get(rec["qid"], d) is None:
                    todo.append((rec["qid"], d))
        if todo:
            for (qid, d), v in zip(todo, reranker.score([(questions[qid], texts[d]) for qid, d in todo])):
                cache.put(qid, d, v)
        _mps_empty_cache()
        if (g // group) % 10 == 0 or g + group >= len(recs):
            cache.save()
            done = min(g + group, len(recs))
            rate = (reranker.n_pairs - n0) / max(1e-9, time.perf_counter() - t_pass)
            mem_samples.append({"at": f"scoring q{done}", **_mps_memory()})
            print(f"      scoring {done}/{len(recs)} queries | {rate:.2f} pairs/s | "
                  f"cache hits {cache.hits} | {_mps_memory()}", flush=True)
    cache.save()
    scoring_pass_s = time.perf_counter() - t_pass

    # ---- ranking: raw logits, stable sort, ties keep candidate order.
    for rec in recs:
        score = {d: cache.scores.get((rec["qid"], d)) for a in RERANKED for d in rec["cand"][a]}
        if any(v is None for v in score.values()):
            raise RuntimeError(f"{rec['qid']}: unscored candidate after the scoring pass")
        rec["rerank_scores"] = {d: round(v, 5) for d, v in score.items()}
        rec["n_cand"] = {a: len(rec["cand"][a]) for a in ARMS}
        rec["final"] = {}
        for a in ARMS:
            cand = rec["cand"][a]
            if a in RERANKED:
                idx = sorted(range(len(cand)), key=lambda i: (-score[cand[i]], i))
                rec["final"][a] = [cand[i] for i in idx[:10]]
            else:
                rec["final"][a] = list(cand[:10])
    comp = [abs(v - cache.scores[(q, d)]) for (q, d), v in timing_scores.items()]
    stage.update({
        "rerank_model": args.rerank_model, "rerank_device": device,
        "rerank_dtype": args.rerank_dtype, "rerank_dtype_actual": dtype_actual,
        "rerank_batch_size": reranker.batch, "rerank_buckets": list(reranker.buckets),
        "rerank_max_length": RERANK_MAX_LENGTH, "rerank_scores": "raw logits",
        "rerank_implementation": "fixed-shape bucketed batching (Amendment 2)",
        "rerank_warm_up": warm, "rerank_smoke_parity": parity,
        "rerank_timing_queries": n_timing, "rerank_timing_pass_s": round(timing_pass_s, 1),
        "rerank_scoring_pass_s": round(scoring_pass_s, 1),
        "rerank_pairs_scored": reranker.n_pairs, "rerank_batches": reranker.n_batches,
        "rerank_padded_rows": reranker.n_padded_rows,
        "rerank_cache": {"path": str(args.rerank_cache) if args.rerank_cache else None,
                         "loaded": cache.loaded, "hits": cache.hits, "size": len(cache.scores)},
        "rerank_batch_composition_abs_diff": {"max": round(max(comp), 6) if comp else None,
                                              "mean": round(float(np.mean(comp)), 6) if comp else None,
                                              "n_pairs": len(comp)},
        "rerank_memory_samples": mem_samples,
    })
    del ce, reranker
    gc.collect()
    _mps_empty_cache()
    return stage


# ---- analysis ----------------------------------------------------------------

def analyse(args, stage, qtype_of, hop_of):
    recs = stage["records"]
    for r in recs:
        r["metrics"] = {a: {f"ce@{k}": complete_evidence(r["final"][a], r["gold"], k)
                            for k in KS} | {f"r@{k}": recall_at_k(r["final"][a], r["gold"], k)
                                             for k in KS}
                        for a in ARMS}
        r["qtype"] = qtype_of.get(r["qid"])
        if hop_of:
            r["hop"] = hop_of.get(r["qid"])

    def agg(rows, arm, key):
        return round(float(np.mean([r["metrics"][arm][key] for r in rows])), 4) if rows else None

    def contrast(rows, x, y, key):
        return paired_bootstrap([r["metrics"][x][key] for r in rows],
                                [r["metrics"][y][key] for r in rows],
                                n_boot=args.boot, seed=args.seed)

    metric_keys = [f"ce@{k}" for k in KS] + [f"r@{k}" for k in KS]
    contrasts = [("A6", "A4"), ("A7", "A4"), ("A8", "A4"), ("A6", "A8"),
                 ("A4", "A3"), ("A5", "A1"), ("A5", "A3"), ("A6", "A5"), ("A3", "A1")]
    slices = {"all": recs}
    comp = [r for r in recs if r.get("qtype") == "comparison"]
    if comp:
        slices["comparison-type questions"] = comp
    few = [r for r in recs if r["n_seeds"] < 2]
    if few:
        slices["fewer than two hubmesh seeds"] = few
    if hop_of:
        for h in sorted({r["hop"] for r in recs if r.get("hop")}):
            slices[f"{h}-hop"] = [r for r in recs if r.get("hop") == h]

    primary = contrast(recs, "A6", "A4", "ce@5")
    is_primary_config = (args.dataset == "2wiki" and "bge-m3" in args.embed_model)
    guard = contrast(comp, "A6", "A4", "ce@5") if comp else None

    def latency(arm):
        rows = [t for r in recs if (t := arm_stage_times(r, arm)) is not None]
        return {**compose_latency(rows, STAGES_USED[arm]), "n_queries": len(rows)}

    timing_rows = [r for r in recs if r["stage_times"].get("rerank_by_arm")]
    cand_counts = {a: [len(r["cand"][a]) for r in recs] for a in sorted(RERANKED)}
    out = {
        "manifest": build_manifest(
            harness=__file__, protocol="benchmarks/PROTOCOL_complementarity.md",
            protocol_sha256=protocol_sha256(), dataset=args.dataset, n=args.n,
            seed=args.seed, representation="title+body",
            embed_model=stage["embed_model"], embed_device=stage["embed_device"],
            embed_dtype=stage.get("embed_dtype", "fp32"),
            embed_batch_size=stage["embed_batch_size"],
            embed_max_seq_len=stage.get("embed_max_seq_len"),
            stage_file={k: stage.get(k) for k in ("dataset", "n", "seed", "qids_sha256", "pool_sha256",
                                                  "protocol_sha256", "git_commit")},
            stage_validation=stage.get("resume_validation"),
            reference_arms_extended=stage.get("reference_arms_extended"),
            gold_refresh=stage.get("gold_refresh"),
            rerank_model=stage.get("rerank_model"), rerank_device=stage.get("rerank_device"),
            rerank_dtype=stage.get("rerank_dtype", "fp32"),
            rerank_dtype_actual=stage.get("rerank_dtype_actual"),
            rerank_batch_size=stage.get("rerank_batch_size"),
            rerank_buckets=stage.get("rerank_buckets"), rerank_max_length=stage.get("rerank_max_length"),
            rerank_scores=stage.get("rerank_scores"),
            rerank_implementation=stage.get("rerank_implementation"),
            rerank_warm_up=stage.get("rerank_warm_up"),
            rerank_smoke_parity=stage.get("rerank_smoke_parity"),
            rerank_timing_queries=stage.get("rerank_timing_queries"),
            rerank_timing_pass_s=stage.get("rerank_timing_pass_s"),
            rerank_scoring_pass_s=stage.get("rerank_scoring_pass_s"),
            rerank_pairs_scored=stage.get("rerank_pairs_scored"),
            rerank_batches=stage.get("rerank_batches"), rerank_padded_rows=stage.get("rerank_padded_rows"),
            rerank_cache=stage.get("rerank_cache"),
            rerank_batch_composition_abs_diff=stage.get("rerank_batch_composition_abs_diff"),
            rerank_memory_samples=stage.get("rerank_memory_samples"),
            hardware=f"{platform.machine()} {platform.platform()}",
            mps_high_watermark_ratio=os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
            mps_low_watermark_ratio=os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO"),
            rrf_k=RRF_K, fuse_topn=TOPN_FUSE, base_n=BASE_N, hub_n=HUB_N,
            rerank_n=RERANK_N, ref_n=REF_N, boot=args.boot),
        "arms": ARM_DESC,
        "corpus": {k: stage[k] for k in ("pool", "kg_nodes", "kg_edges", "kg_build_s",
                                         "embed_pool_s")},
        "n": len(recs),
        "primary": {
            "contrast": "A6 - A4, complete-evidence@5",
            "is_primary_configuration": is_primary_config,
            "result": primary,
            "decision": decide(primary) if is_primary_config else
                        f"(secondary configuration) {decide(primary)}",
            "regression_guard": {
                "slice": "comparison-type questions (question-type slice; not a claim about simple questions)",
                "n": len(comp), "A6-A4 ce@5": guard,
                "rule": "point estimate >= -1.0",
                "met": (guard["mean"] >= -1.0) if guard else None,
            },
        },
        "diagnostics": {
            "frac_queries_with_ge2_seeds": round(float(np.mean([r["n_seeds"] >= 2 for r in recs])), 4),
            "mean_overlap_hybrid30_hub20": round(float(np.mean([r["overlap_hybrid30_hub20"] for r in recs])), 2),
            "frac_gold_docs_without_entities": round(
                float(sum(r["gold_docs_without_entities"] for r in recs)
                      / max(1, sum(r["n_gold"] - r.get("n_gold_unresolved", 0) for r in recs))), 4),
            "gold_ids_unresolved": int(sum(r.get("n_gold_unresolved", 0) for r in recs)),
            "queries_with_unresolved_gold": int(sum(1 for r in recs if r.get("n_gold_unresolved", 0))),
            "candidate_counts": {a: {"min": int(min(v)), "mean": round(float(np.mean(v)), 2),
                                     "queries_below_50": int(sum(1 for c in v if c < RERANK_N))}
                                 for a, v in cand_counts.items()},
            "mean_candidates_A6": round(float(np.mean(cand_counts["A6"])), 2),
        },
        "results": {name: {a: {k: agg(rows, a, k) for k in metric_keys} for a in ARMS}
                    for name, rows in slices.items()},
        "contrasts": {name: {f"{x}-{y}": {k: contrast(rows, x, y, k) for k in metric_keys}
                             for x, y in contrasts}
                      for name, rows in slices.items()},
        "latency_composed_estimate": {
            "note": ("per-query sums of measured stage times, then median/p95; "
                     "excludes model switching, cold start and process/request "
                     "overhead — an estimate of a both-models-resident deployment. "
                     "Rerank time is measured per arm on the first "
                     f"{stage.get('rerank_timing_queries')} queries (Amendment 2); the reranked "
                     "arms' composed figures are over those queries (n_queries)"),
            "stages_per_arm": STAGES_USED,
            "stage_medians_ms": {s: round(float(np.median([r["stage_times"].get(s, 0.0)
                                                            for r in recs])) * 1000, 2)
                                 for s in ("embed", "dense", "bm25", "rrf", "hubmesh")},
            "rerank_median_ms_by_arm": {a: round(float(np.median([r["stage_times"]["rerank_by_arm"][a]
                                                                  for r in timing_rows])) * 1000, 2)
                                        for a in sorted(RERANKED)} if timing_rows else None,
            "arms": {a: latency(a) for a in ARMS},
        },
        "per_query": recs,
    }
    a6 = out["latency_composed_estimate"]["arms"]["A6"]["p95_ms"]
    a4 = out["latency_composed_estimate"]["arms"]["A4"]["p95_ms"]
    out["latency_composed_estimate"]["p95_ratio_A6_over_A4"] = (
        round(a6 / a4, 3) if a6 and a4 else None)
    out["latency_composed_estimate"]["rule_met_le_1.5"] = (
        (a6 / a4 <= 1.5) if a6 and a4 else None)
    return out


# ---- main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["2wiki", "hotpotqa", "musique"], required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embed-model", default="BAAI/bge-m3")
    ap.add_argument("--embed-device", default=None)
    ap.add_argument("--embed-batch-size", type=int, default=8)
    ap.add_argument("--embed-max-seq-len", type=int, default=512)
    ap.add_argument("--embed-dtype", choices=["fp32", "fp16"], default="fp32",
                    help="fp16 halves the model's memory on Metal (Amendment 1)")
    ap.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--rerank-device", default=None)
    ap.add_argument("--rerank-dtype", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--rerank-batch-size", type=int, default=8)
    ap.add_argument("--rerank-buckets", default=",".join(map(str, RERANK_BUCKETS)),
                    help="padded sequence lengths the reranker sees (Amendment 2)")
    ap.add_argument("--rerank-timing-queries", type=int, default=100,
                    help="queries on which every reranked arm is timed separately (Amendment 2)")
    ap.add_argument("--rerank-cache", default=None,
                    help="(qid, doc) -> logit cache shared by a dataset's configurations "
                         "(default: <out dir>/rerank_cache_<dataset>_n<n>_s<seed>_<device>_<dtype>.json)")
    ap.add_argument("--rerank-parity-tolerance", type=float, default=0.05,
                    help="abort if the fixed-shape path differs from the library path by more (logits)")
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--stage", choices=["all", "retrieve", "rerank"], default="all")
    ap.add_argument("--stage-file", default=None,
                    help="intermediate candidates/timings (default: <out>.stage.json); never rewritten by stage K")
    ap.add_argument("--no-print-results", action="store_true",
                    help="write the result file without printing the metrics (examine once, later)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.rerank_buckets = tuple(int(b) for b in str(args.rerank_buckets).split(","))
    out_path = Path(args.out)
    stage_path = Path(args.stage_file) if args.stage_file else out_path.with_suffix(".stage.json")

    print(f"[1/4] Loading {args.dataset} n={args.n} seed={args.seed}")
    examples, ids, texts, titles, gold_of, qtype_of, hop_of = load_data(args.dataset, args.n, args.seed)
    pool = set(ids)
    print(f"      pool {len(ids)} paragraphs, {len(examples)} questions, "
          f"{sum(1 for ex in examples if all(g in pool for g in gold_of[ex.qid]))} with every gold paragraph in the pool")
    questions = {ex.qid: ex.question for ex in examples}

    if args.stage in ("all", "retrieve"):
        print(f"[2/4] Stage R: {args.embed_model} + KG + BM25 + hubmesh")
        stage = stage_retrieve(args, examples, ids, texts, gold_of)
        stage_path.parent.mkdir(parents=True, exist_ok=True)
        stage_path.write_text(json.dumps(stage))
        print(f"      stage file -> {stage_path}")
        if args.stage == "retrieve":
            return
    else:
        stage = json.loads(stage_path.read_text())
        expected = {"dataset": args.dataset, "n": args.n, "seed": args.seed, "embed_model": args.embed_model}
        stage["resume_validation"] = validate_stage(stage, expected, [ex.qid for ex in examples], ids)
        print(f"      stage file {stage_path} validated: {stage['resume_validation']}")
        if any(len(r["cand"][a]) < REF_N for r in stage["records"] for a in ("A1", "A2", "A3", "A5")):
            bm25_top = bm25_rankings(examples, ids, texts)
            n_ext = extend_reference_arms(stage["records"], bm25_top)
            stage["reference_arms_extended"] = {
                "lists_extended": n_ext, "to": REF_N,
                "how": "A1 = A7 prefix, A3 = A4 prefix, A5 = A8 prefix, A2 = BM25 recomputed; "
                       "stored 5-prefixes verified"}
            print(f"      reference arms extended to {REF_N}: {n_ext} lists (stored prefixes verified)")
        stage["gold_refresh"] = refresh_gold(stage["records"], gold_of, ids)
        print(f"      gold refreshed to the full lists: {stage['gold_refresh']}")

    if args.rerank_cache is None:
        args.rerank_cache = out_path.parent / (f"rerank_cache_{args.dataset}_n{args.n}_s{args.seed}_"
                                               f"{args.rerank_device or 'auto'}_{args.rerank_dtype}.json")
    print(f"[3/4] Stage K: {args.rerank_model}")
    stage = stage_rerank(args, stage, texts, questions)

    print("[4/4] Analysis (examined once)")
    out = analyse(args, stage, qtype_of, hop_of)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    lat = out["latency_composed_estimate"]
    print(f"      rerank median ms by arm: {lat['rerank_median_ms_by_arm']} "
          f"(timing queries: {stage.get('rerank_timing_queries')})")
    if args.no_print_results:
        print(f"wrote {out_path} (metrics not printed: --no-print-results)")
        return
    p = out["primary"]
    print(f"primary A6-A4 ce@5: {p['result']} -> {p['decision']}")
    print(f"guard {p['regression_guard']['slice']}: n={p['regression_guard']['n']} "
          f"{p['regression_guard']['A6-A4 ce@5']} met={p['regression_guard']['met']}")
    for a in ARMS:
        r = out["results"]["all"][a]
        print(f"  {a:<3} ce@5 {r['ce@5']:.3f}  ce@10 {r['ce@10']:.3f}  r@5 {r['r@5']:.3f}  r@10 {r['r@10']:.3f}"
              f"  | composed p95 {lat['arms'][a]['p95_ms']} ms (n={lat['arms'][a]['n_queries']})")
    print("diagnostics:", out["diagnostics"])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
