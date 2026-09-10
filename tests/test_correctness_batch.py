"""Regression tests for the 2026-09 external-review correctness batch:

  1. corpus names cannot escape the storage root
  2. packed context and returned sources agree exactly
  3. chunker output flows directly into from_documents(embed=...)
  4. invalid chunk arguments raise instead of hanging/skipping
  5. embedding fingerprint recorded and enforced on load
  6. corpus rebuild is atomic — a failed rebuild leaves the old
     generation intact and no temp debris
"""
import json

import numpy as np
import networkx as nx
import pytest

from hubmesh import Document, Planner
from hubmesh.adapters import InMemoryStore
from hubmesh.chunking import chunk_by_chars, chunk_by_sentences, chunk_documents
from hubmesh.corpus import CorpusManager
from hubmesh.kg import EntityKG


def stub_embed(text: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.normal(size=16).astype(np.float32)
    return v / np.linalg.norm(v)


def minimal_kg(doc_ids):
    g = nx.Graph()
    for d in doc_ids:
        g.add_node(f"doc:{d}", kind="doc")
    return EntityKG(graph=g,
                    doc_to_entities={d: set() for d in doc_ids},
                    entity_to_docs={},
                    entity_canonical_to_node={},
                    entity_node_to_label={},
                    alias_to_node={})


def build_simple(mgr, name, texts=("alpha text", "beta text")):
    return mgr.build(name, list(texts), kg=minimal_kg(
        [str(i) for i in range(len(texts))]))


def live_meta_path(root, name):
    cdir = root / name
    gen = (cdir / "CURRENT").read_text().strip()
    return cdir / gen / "meta.json"


# ---- 1. path containment -------------------------------------------------

class TestCorpusNameContainment:
    @pytest.mark.parametrize("bad", [
        "../escaped", "a/b", "/abs/path", "..", ".hidden", "", "a" * 65,
        "a\\b", "name/../..",
    ])
    def test_rejects_non_identifier_names(self, tmp_path, bad):
        mgr = CorpusManager(root=tmp_path, embed=stub_embed)
        with pytest.raises((ValueError, FileNotFoundError)):
            build_simple(mgr, bad)
        # nothing may exist outside the root
        assert not (tmp_path.parent / "escaped").exists()

    def test_load_rejects_traversal(self, tmp_path):
        mgr = CorpusManager(root=tmp_path, embed=stub_embed)
        with pytest.raises(ValueError):
            mgr.load("../whatever")

    def test_valid_name_roundtrips(self, tmp_path):
        mgr = CorpusManager(root=tmp_path, embed=stub_embed,
                            embed_identity="stub-16d")
        meta = build_simple(mgr, "good_name-1.2")
        assert meta["n_docs"] == 2
        store, kg = mgr.load("good_name-1.2")
        assert len(store.all_ids()) == 2


# ---- 2. context/sources agreement ---------------------------------------

class TestContextSourceAgreement:
    def make_planner(self, n=12):
        docs = [Document(id=str(i), text=f"document number {i} " * 20,
                         vector=stub_embed(f"doc {i}"))
                for i in range(n)]
        return Planner(store=InMemoryStore(docs, k=4))

    def test_context_docs_equal_sources(self):
        planner = self.make_planner()
        for top_k in (1, 3, 5):
            res = planner.retrieve(stub_embed("query"), top_k=top_k,
                                   budget_tokens=100_000)
            n_blocks = res.context.count("[") if res.context else 0
            assert len(res.sources) <= top_k
            # every numbered block corresponds to a returned source
            assert n_blocks == len(res.sources), (
                f"top_k={top_k}: context has {n_blocks} documents but "
                f"sources lists {len(res.sources)}")


# ---- 3+4. chunking -------------------------------------------------------

class TestChunking:
    def test_chunk_output_flows_into_store(self):
        chunks = chunk_documents([("src", "Alpha beta. " * 200)],
                                 strategy="chars", chunk_chars=200,
                                 overlap_chars=20)
        assert chunks and all(c.vector is None for c in chunks)
        store = InMemoryStore.from_documents(chunks, embed=stub_embed)
        assert len(store.all_ids()) == len(chunks)

    def test_unembedded_document_without_embed_raises(self):
        chunks = chunk_documents([("src", "Alpha beta. " * 50)],
                                 strategy="chars")
        with pytest.raises(ValueError, match="no vector"):
            InMemoryStore.from_documents(chunks)

    @pytest.mark.parametrize("kwargs", [
        {"chunk_chars": 0}, {"chunk_chars": -5}, {"overlap_chars": -1},
    ])
    def test_invalid_char_args_raise(self, kwargs):
        with pytest.raises(ValueError):
            chunk_by_chars("s", "some text " * 100, **kwargs)

    @pytest.mark.parametrize("kwargs", [
        {"target_tokens": 0}, {"overlap_sentences": -1},
    ])
    def test_invalid_sentence_args_raise(self, kwargs):
        with pytest.raises(ValueError):
            chunk_by_sentences("s", "One. Two. Three.", **kwargs)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy"):
            chunk_documents([("s", "text")], strategy="bogus")


# ---- 5. embedding fingerprint --------------------------------------------

class TestEmbeddingFingerprint:
    def test_identity_recorded_and_mismatch_rejected(self, tmp_path):
        a = CorpusManager(root=tmp_path, embed=stub_embed,
                          embed_identity="model-A")
        build_simple(a, "corp")
        meta = json.loads(live_meta_path(tmp_path, "corp").read_text())
        assert meta["embedding"]["identity"] == "model-A"
        assert meta["embedding"]["dim"] == 16

        b = CorpusManager(root=tmp_path, embed=stub_embed,
                          embed_identity="model-B")
        with pytest.raises(ValueError, match="model-A"):
            b.load("corp")

    def test_matching_identity_loads(self, tmp_path):
        a = CorpusManager(root=tmp_path, embed=stub_embed,
                          embed_identity="model-A")
        build_simple(a, "corp")
        again = CorpusManager(root=tmp_path, embed=stub_embed,
                              embed_identity="model-A")
        store, _ = again.load("corp")
        assert len(store.all_ids()) == 2

    def test_unknown_current_identity_warns(self, tmp_path):
        a = CorpusManager(root=tmp_path, embed=stub_embed,
                          embed_identity="model-A")
        build_simple(a, "corp")
        b = CorpusManager(root=tmp_path, embed=stub_embed)  # no identity
        with pytest.warns(UserWarning, match="cannot be verified"):
            b.load("corp")

    def test_future_format_version_rejected(self, tmp_path):
        a = CorpusManager(root=tmp_path, embed=stub_embed,
                          embed_identity="m")
        build_simple(a, "corp")
        meta_path = live_meta_path(tmp_path, "corp")
        meta = json.loads(meta_path.read_text())
        meta["format_version"] = 99
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="format_version"):
            a.load("corp")


# ---- 6. atomic rebuild + generation consistency --------------------------

class TestAtomicRebuild:
    def test_staging_failure_preserves_old_generation(self, tmp_path,
                                                      monkeypatch):
        mgr = CorpusManager(root=tmp_path, embed=stub_embed,
                            embed_identity="m")
        build_simple(mgr, "corp", texts=("original one", "original two"))

        import hubmesh.corpus as corpus_mod

        def boom(*a, **k):
            raise IOError("simulated crash mid-write")
        monkeypatch.setattr(corpus_mod.np, "savez_compressed", boom)
        with pytest.raises(IOError):
            build_simple(mgr, "corp", texts=("replacement",))
        monkeypatch.undo()

        store, _ = mgr.load("corp")
        texts = {store.get(i).text for i in store.all_ids()}
        assert texts == {"original one", "original two"}
        staging = list((tmp_path / "corp").glob("*.staging"))
        assert staging == []
        assert set(mgr.list()) == {"corp"}

    def test_publish_step_failure_preserves_live_corpus(self, tmp_path,
                                                        monkeypatch):
        """The reviewer's scenario: crash AT the publish step. With the
        pointer design the corpus must remain listed and loadable with
        its old content — not vanish into a hidden backup."""
        import os as os_mod
        mgr = CorpusManager(root=tmp_path, embed=stub_embed,
                            embed_identity="m")
        build_simple(mgr, "corp", texts=("original one", "original two"))

        real_replace = os_mod.replace

        def boom(src, dst, *a, **k):
            if str(dst).endswith("CURRENT"):
                raise OSError("simulated crash during publish")
            return real_replace(src, dst, *a, **k)
        monkeypatch.setattr(os_mod, "replace", boom)
        with pytest.raises(OSError):
            build_simple(mgr, "corp", texts=("replacement",))
        monkeypatch.undo()

        assert set(mgr.list()) == {"corp"}          # never disappeared
        store, _ = mgr.load("corp")
        texts = {store.get(i).text for i in store.all_ids()}
        assert texts == {"original one", "original two"}
        # a later rebuild self-heals the orphan generation
        build_simple(mgr, "corp", texts=("new gen",))
        store, _ = mgr.load("corp")
        assert {store.get(i).text for i in store.all_ids()} == {"new gen"}

    def test_reader_survives_multiple_rebuilds_within_retention(self,
                                                                tmp_path):
        """Round-3 repro: a reader that resolved gen G0, then two rebuilds
        complete, must still find G0's files (age-based retention)."""
        mgr = CorpusManager(root=tmp_path, embed=stub_embed,
                            embed_identity="m")
        build_simple(mgr, "corp", texts=("gen one text",))
        cdir = tmp_path / "corp"
        gen1 = (cdir / "CURRENT").read_text().strip()

        build_simple(mgr, "corp", texts=("gen two text",))
        build_simple(mgr, "corp", texts=("gen three text",))
        # TWO rebuilds later, G0 is still fully readable inside the window
        assert "gen one text" in (cdir / gen1 / "docs.jsonl").read_text()
        assert (cdir / gen1 / "vectors.npz").exists()

    def test_expired_generations_are_pruned(self, tmp_path):
        mgr = CorpusManager(root=tmp_path, embed=stub_embed,
                            embed_identity="m", retention_seconds=0.0)
        build_simple(mgr, "corp", texts=("one",))
        build_simple(mgr, "corp", texts=("two",))
        build_simple(mgr, "corp", texts=("three",))
        cdir = tmp_path / "corp"
        gens = [p.name for p in cdir.glob("gen-*")]
        assert gens == [(cdir / "CURRENT").read_text().strip()]
        store, _ = mgr.load("corp")
        assert {store.get(i).text for i in store.all_ids()} == {"three"}

    def test_concurrent_writers_never_break_current(self, tmp_path,
                                                    monkeypatch):
        """Round-3 repro: overlapping writers must serialize; afterwards
        CURRENT must name an existing, loadable generation and list()
        must show the corpus."""
        import threading
        import time as time_mod
        import hubmesh.corpus as corpus_mod

        mgr = CorpusManager(root=tmp_path, embed=stub_embed,
                            embed_identity="m", retention_seconds=0.0)
        build_simple(mgr, "corp", texts=("seed",))

        real_savez = corpus_mod.np.savez_compressed

        def slow_savez(*a, **k):
            time_mod.sleep(0.25)
            return real_savez(*a, **k)
        monkeypatch.setattr(corpus_mod.np, "savez_compressed", slow_savez)

        errors = []

        def writer(text):
            try:
                m = CorpusManager(root=tmp_path, embed=stub_embed,
                                  embed_identity="m", retention_seconds=0.0)
                build_simple(m, "corp", texts=(text,))
            except Exception as e:      # pragma: no cover
                errors.append(e)

        a = threading.Thread(target=writer, args=("from writer A",))
        b = threading.Thread(target=writer, args=("from writer B",))
        a.start(); time_mod.sleep(0.05); b.start()
        a.join(); b.join()
        monkeypatch.undo()

        assert errors == []
        cdir = tmp_path / "corp"
        current = (cdir / "CURRENT").read_text().strip()
        assert (cdir / current).is_dir()            # pointer target exists
        assert set(mgr.list()) == {"corp"}
        store, _ = mgr.load("corp")
        texts = {store.get(i).text for i in store.all_ids()}
        assert texts in ({"from writer A"}, {"from writer B"})

    def test_old_live_corpus_first_rebuild_protects_reader(self, tmp_path):
        """Round-4 repro: a generation that has been CURRENT for a long
        time must still get the FULL grace period when replaced — the
        window runs from retirement, not creation."""
        import os as os_mod
        import time as time_mod
        mgr = CorpusManager(root=tmp_path, embed=stub_embed,
                            embed_identity="m")          # default 900s
        build_simple(mgr, "corp", texts=("old live text",))
        cdir = tmp_path / "corp"
        gen0 = (cdir / "CURRENT").read_text().strip()
        # simulate a corpus that has been live for two hours
        two_hours_ago = time_mod.time() - 7200
        os_mod.utime(cdir / gen0, (two_hours_ago, two_hours_ago))

        build_simple(mgr, "corp", texts=("replacement",))
        # the aged generation survives with a fresh retirement marker —
        # a reader paused mid-load still finds every file
        assert (cdir / gen0 / "docs.jsonl").exists()
        assert (cdir / gen0 / "vectors.npz").exists()
        retired_at = float((cdir / gen0 / ".retired").read_text())
        assert time_mod.time() - retired_at < 60

    def test_pruned_only_after_retirement_window_expires(self, tmp_path):
        import time as time_mod
        mgr = CorpusManager(root=tmp_path, embed=stub_embed,
                            embed_identity="m")          # default 900s
        build_simple(mgr, "corp", texts=("one",))
        cdir = tmp_path / "corp"
        gen0 = (cdir / "CURRENT").read_text().strip()
        build_simple(mgr, "corp", texts=("two",))
        assert (cdir / gen0).exists()                    # inside window
        # simulate the window elapsing since RETIREMENT
        (cdir / gen0 / ".retired").write_text(
            repr(time_mod.time() - 3600))
        build_simple(mgr, "corp", texts=("three",))
        assert not (cdir / gen0).exists()                # expired -> pruned

    def test_missing_retirement_marker_is_conservative(self, tmp_path):
        """Interrupted bookkeeping (crash between publish and marker):
        an unmarked retired generation must NOT be treated as expired —
        the clock starts when the miss is discovered."""
        import os as os_mod
        import time as time_mod
        mgr = CorpusManager(root=tmp_path, embed=stub_embed,
                            embed_identity="m")
        build_simple(mgr, "corp", texts=("one",))
        cdir = tmp_path / "corp"
        gen0 = (cdir / "CURRENT").read_text().strip()
        build_simple(mgr, "corp", texts=("two",))
        # simulate the crash: marker never written, dir looks ancient
        (cdir / gen0 / ".retired").unlink()
        old = time_mod.time() - 7200
        os_mod.utime(cdir / gen0, (old, old))

        build_simple(mgr, "corp", texts=("three",))
        assert (cdir / gen0).exists()                    # NOT deleted
        assert (cdir / gen0 / ".retired").exists()       # clock started
        # once that fresh window elapses, it is collectable
        (cdir / gen0 / ".retired").write_text(
            repr(time_mod.time() - 3600))
        build_simple(mgr, "corp", texts=("four",))
        assert not (cdir / gen0).exists()

    def test_legacy_flat_files_survive_migration(self, tmp_path):
        """Round-3: legacy flat files are never deleted — a legacy reader's
        lifetime is unknowable, and they are inert once CURRENT exists."""
        mgr = CorpusManager(root=tmp_path, embed=stub_embed,
                            embed_identity="m", retention_seconds=0.0)
        build_simple(mgr, "corp", texts=("flat text",))
        cdir = tmp_path / "corp"
        gen = (cdir / "CURRENT").read_text().strip()
        for f in list((cdir / gen).iterdir()):
            f.rename(cdir / f.name)
        (cdir / gen).rmdir()
        (cdir / "CURRENT").unlink()

        build_simple(mgr, "corp", texts=("migrated text",))
        assert (cdir / "docs.jsonl").exists()       # legacy files intact
        assert "flat text" in (cdir / "docs.jsonl").read_text()
        store, _ = mgr.load("corp")                 # new layout wins
        assert {store.get(i).text for i in store.all_ids()} == {
            "migrated text"}

    def test_legacy_flat_layout_still_loads(self, tmp_path):
        mgr = CorpusManager(root=tmp_path, embed=stub_embed,
                            embed_identity="m")
        build_simple(mgr, "corp", texts=("flat text",))
        cdir = tmp_path / "corp"
        # flatten: simulate a pre-generation corpus
        gen = (cdir / "CURRENT").read_text().strip()
        for f in (cdir / gen).iterdir():
            f.rename(cdir / f.name)
        (cdir / gen).rmdir()
        (cdir / "CURRENT").unlink()
        store, _ = mgr.load("corp")
        assert {store.get(i).text for i in store.all_ids()} == {"flat text"}


# ---- default-path fingerprint (reviewer round 2) --------------------------

class TestDefaultPathFingerprint:
    """The identity must survive lazy default-model initialization — the
    round-2 review reproduced identity=null via exactly this path."""

    @pytest.fixture
    def stub_st(self, monkeypatch):
        import hubmesh.entity_linker as el
        monkeypatch.setattr(
            el, "make_st_embedder",
            lambda model: (lambda texts: [stub_embed(t) for t in texts]))

    def test_default_build_records_env_model(self, tmp_path, stub_st,
                                             monkeypatch):
        monkeypatch.setenv("HUBMESH_EMBED_MODEL", "model-A")
        mgr = CorpusManager(root=tmp_path)          # default lazy path
        build_simple(mgr, "corp")
        meta = json.loads(live_meta_path(tmp_path, "corp").read_text())
        assert meta["embedding"]["identity"] == "model-A"

    def test_default_load_with_other_model_rejected(self, tmp_path, stub_st,
                                                    monkeypatch):
        monkeypatch.setenv("HUBMESH_EMBED_MODEL", "model-A")
        build_simple(CorpusManager(root=tmp_path), "corp")

        monkeypatch.setenv("HUBMESH_EMBED_MODEL", "model-B")
        other = CorpusManager(root=tmp_path)
        with pytest.raises(ValueError, match="model-A"):
            other.load("corp")

    def test_warmed_default_model_still_rejects_mismatch(self, tmp_path,
                                                         stub_st,
                                                         monkeypatch):
        """Round-2 repro: initializing the model BEFORE loading must not
        downgrade the rejection to a warning."""
        monkeypatch.setenv("HUBMESH_EMBED_MODEL", "model-A")
        build_simple(CorpusManager(root=tmp_path), "corp")

        monkeypatch.setenv("HUBMESH_EMBED_MODEL", "model-B")
        other = CorpusManager(root=tmp_path)
        other._get_embed()                           # warm model B first
        with pytest.raises(ValueError, match="model-A"):
            other.load("corp")


class TestDimensionEnforcement:
    def test_meta_vector_dim_mismatch_rejected(self, tmp_path):
        mgr = CorpusManager(root=tmp_path, embed=stub_embed,
                            embed_identity="m")
        build_simple(mgr, "corp")
        cdir = tmp_path / "corp"
        gen = (cdir / "CURRENT").read_text().strip()
        meta_path = cdir / gen / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["embedding"]["dim"] = 999
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="999"):
            mgr.load("corp")
