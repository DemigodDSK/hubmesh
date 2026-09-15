"""Reproducibility manifest for benchmark result files.

Every benchmark/research script that writes results should embed
`build_manifest(...)` so a number can be traced to the exact code,
environment, models, dataset slice, and document representation that
produced it (external review: older and newer tables could not be
reconciled because none of this was recorded).

Identity of the implementation is recorded two ways (external review,
2026-09-14: a commit hash plus `git_dirty=true` does not identify the
code that ran):
  * `git_commit` / `git_dirty` / `git_dirty_paths` — dirtiness over the
    WHOLE tree, not just `src/`;
  * `src_sha256` — content hash of every `src/hubmesh/**/*.py`;
    `harness_sha256` — content hash of the script that produced the
    file (`sys.argv[0]` unless `harness=` is passed); and
    `benchmarks_sha256` — content hash of every `benchmarks/**/*.py`,
    which covers the imported helpers (loaders, `structural_only.py`,
    `hippo_style.py`, this module) a runner's own hash does not.
Callers record the embedding device and batch size as extra fields.
"""
from __future__ import annotations
import hashlib
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _version(mod: str) -> str | None:
    try:
        m = __import__(mod)
        return getattr(m, "__version__", None)
    except Exception:
        return None


def _tree_sha256(paths: list[Path], base: Path) -> str:
    """Order-independent content hash of a set of files: each file's
    path relative to `base` and its bytes, in sorted path order."""
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p.relative_to(base)).encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _file_sha256(p: Path) -> str | None:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def build_manifest(harness: str | None = None, **extra) -> dict:
    """Commit identity, dirty flag, source/harness content hashes,
    library/model versions, platform, and any caller-supplied fields
    (dataset, n, seed, representation, planner config, embedding model,
    ...)."""
    sys.path.insert(0, str(ROOT / "src"))
    dirty = (_git("status", "--porcelain") or "").splitlines()
    src_files = sorted((ROOT / "src" / "hubmesh").rglob("*.py"))
    bench_files = sorted((ROOT / "benchmarks").rglob("*.py"))
    harness_path = Path(harness) if harness else Path(sys.argv[0] or "")
    try:
        harness_rel = str(harness_path.resolve().relative_to(ROOT))
    except (ValueError, OSError):
        harness_rel = str(harness_path)
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(dirty),
        "git_dirty_paths": dirty[:20],
        "src_sha256": _tree_sha256(src_files, ROOT) if src_files else None,
        "benchmarks_sha256": _tree_sha256(bench_files, ROOT) if bench_files else None,
        "harness": harness_rel,
        "harness_sha256": _file_sha256(harness_path) if harness_path.name else None,
        "hubmesh_version": _version("hubmesh"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libs": {m: _version(m) for m in
                 ("numpy", "scipy", "networkx", "sentence_transformers",
                  "spacy", "datasets", "torch")},
        **extra,
    }
