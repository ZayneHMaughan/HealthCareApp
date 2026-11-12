"""Build a knowledge-base index.

  python -m healthcoach.retrieval.build --medlineplus mplus_topics_2026-11-11.xml --out models/kb
  python -m healthcoach.retrieval.build --medlineplus ... --curated curated/ --embedder st:BAAI/bge-small-en-v1.5 --out models/kb
"""
from __future__ import annotations
import argparse
from pathlib import Path
from .embed import build_embedder
from .index import Index
from .ingest import load_topic_map, parse_curated_dir, parse_medlineplus
from .retriever import Retriever, coverage_report

DEFAULT_MAP = Path(__file__).parent / "data" / "topic_map.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--medlineplus", help="MedlinePlus Health Topic XML file (from medlineplus.gov/xml.html)")
    ap.add_argument("--curated", help="directory of reviewed Markdown files")
    ap.add_argument("--topic-map", default=str(DEFAULT_MAP))
    ap.add_argument("--embedder", default="none", help="none | hashing | st:<model> | llama:<path.gguf>")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tmap = load_topic_map(a.topic_map)
    chunks = []
    if a.medlineplus:
        c, rep = parse_medlineplus(a.medlineplus, tmap)
        chunks += c
        print(f"MedlinePlus: {rep['n_topics']} topics -> {rep['n_chunks']} chunks (file date {rep['source_date']})")
        print(f"  skipped: {rep['skipped_non_english']} non-English, {rep['skipped_no_summary']} without summary, "
              f"{len(rep['skipped_copyright_marker'])} with copyright marker {rep['skipped_copyright_marker'][:5]}")
        if rep["map_titles_unmatched"]:
            print(f"  topic_map titles NOT found in the file (fix these): {rep['map_titles_unmatched']}")
    if a.curated:
        c = parse_curated_dir(a.curated)
        chunks += c
        print(f"Curated: {len(c)} chunks")
    if not chunks:
        raise SystemExit("nothing to index")

    idx = Index.build(chunks, build_embedder(a.embedder))
    man = idx.save(a.out)
    print(f"Index written to {a.out}: {man['n_chunks']} chunks, embedder={man['embedder']}")

    gaps = [g for g, titles in coverage_report(Retriever(idx)).items() if not titles]
    print(f"\nSymptom groups with NO retrievable passage ({len(gaps)}): {gaps}")
    print("Add curated content or fix topic_map.csv for these; the app falls back to a template for them.")


if __name__ == "__main__":
    main()
