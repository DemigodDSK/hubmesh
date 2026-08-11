"""Generate the markdown corpus for the LightRAG comparison lane.

Format agreed with the LightRAG maintainers in HKUDS/LightRAG#3571:
one pooled HotpotQA paragraph per .md file,

    # {Wikipedia title}

    {paragraph text}

Filenames are filesystem-sanitized; the RAW title is the document id and is
preserved in manifest.json (filename -> title) so scoring stays exact-id.

Outputs, per scope (pilot = N=500 seed 0, same sampler as the ablations;
full = all 7,405 dev questions):

    corpus_<scope>/                 one .md per pooled paragraph
    manifest_<scope>.json           [{title, filename, n_words}]
    questions_<scope>.json          [{qid, question, answer, type, gold_titles}]
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from hotpotqa_loader import load_hotpotqa  # noqa: E402

SCOPES = {"pilot": 500, "full": 7405}


def safe_filename(title: str) -> str:
    """Sanitize a Wikipedia title for the filesystem (AC/DC -> AC_DC)."""
    return re.sub(r'[\\/*?:"<>|]', "_", title).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=SCOPES, default="pilot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(HERE))
    args = ap.parse_args()

    n = SCOPES[args.scope]
    out = Path(args.out_dir)
    corpus_dir = out / f"corpus_{args.scope}"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading HotpotQA dev, n={n}, seed={args.seed}...")
    examples, pool = load_hotpotqa(n_questions=n, seed=args.seed)
    print(f"  {len(examples)} questions, {len(pool)} pooled paragraphs")

    manifest = []
    seen_files: dict[str, str] = {}   # filename -> title, to catch collisions
    for title, text in pool.items():
        fname = safe_filename(title) + ".md"
        if fname in seen_files and seen_files[fname] != title:
            # Two distinct titles sanitizing to the same filename would break
            # the filename->title mapping; disambiguate deterministically.
            stem = fname[:-3]
            i = 2
            while f"{stem}__{i}.md" in seen_files:
                i += 1
            fname = f"{stem}__{i}.md"
        seen_files[fname] = title
        (corpus_dir / fname).write_text(f"# {title}\n\n{text}\n", encoding="utf-8")
        manifest.append({"title": title, "filename": fname,
                         "n_words": len(text.split())})

    (out / f"manifest_{args.scope}.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")

    questions = [
        {"qid": ex.qid, "question": ex.question, "answer": ex.answer,
         "type": ex.qtype, "gold_titles": ex.gold_titles}
        for ex in examples
    ]
    (out / f"questions_{args.scope}.json").write_text(
        json.dumps(questions, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"  wrote {len(manifest)} .md files -> {corpus_dir}")
    print(f"  manifest_{args.scope}.json + questions_{args.scope}.json done")


if __name__ == "__main__":
    main()
