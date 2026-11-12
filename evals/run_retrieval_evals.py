"""Retrieval evaluation: does the right source come back, does the wrong one stay out, and where are the gaps?

  python evals/run_retrieval_evals.py --index models/kb                    # run cases
  python evals/run_retrieval_evals.py --index models/kb --coverage         # list symptom groups with no passage
  python evals/run_retrieval_evals.py --index models/kb --embedder st:BAAI/bge-small-en-v1.5

Case fields: id, intake, tier (optional), expect_titles_any (title substrings, case-insensitive),
forbid_titles, expect_uncovered, expect_first_kind. Exit 1 on any failing case.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from healthcoach.retrieval import Index, Retriever
from healthcoach.retrieval.embed import build_embedder
from healthcoach.retrieval.retriever import coverage_report
from healthcoach.schema import Intake, Tier


def run_cases(retriever: Retriever, cases: list[dict]) -> list[dict]:
    rows = []
    for c in cases:
        tier = Tier[c.get("tier", "see_doctor_soon").upper()]
        res = retriever.retrieve(Intake.from_dict(c["intake"]), tier)
        titles = [p.chunk.title.lower() for p in res.passages]
        problems = []
        want = [w.lower() for w in c.get("expect_titles_any", [])]
        if want and not any(w in t for w in want for t in titles):
            problems.append(f"none of {c['expect_titles_any']} retrieved (got {[p.chunk.title for p in res.passages]})")
        for f in c.get("forbid_titles", []):
            if any(f.lower() in t for t in titles):
                problems.append(f"forbidden title retrieved: {f}")
        if "expect_uncovered" in c and sorted(res.uncovered) != sorted(c["expect_uncovered"]):
            problems.append(f"uncovered {res.uncovered}, expected {c['expect_uncovered']}")
        if c.get("expect_first_kind") and (not res.passages or res.passages[0].chunk.kind != c["expect_first_kind"]):
            problems.append(f"first passage kind != {c['expect_first_kind']}")
        rows.append({"id": c["id"], "problems": problems, "titles": [p.chunk.title for p in res.passages]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--embedder", default="none")
    ap.add_argument("--cases", default="evals/retrieval_cases")
    ap.add_argument("--coverage", action="store_true")
    a = ap.parse_args()
    idx = Index.load(a.index, build_embedder(a.embedder))
    r = Retriever(idx)
    print(f"index: {idx.manifest['n_chunks']} chunks, embedder={idx.manifest['embedder']}, "
          f"age={idx.age_days()} days{' (STALE)' if idx.stale() else ''}")
    if a.coverage:
        gaps = [g for g, t in coverage_report(r).items() if not t]
        print(f"{len(gaps)} symptom groups have no retrievable passage: {gaps}")
        return 0
    p = Path(a.cases)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    cases = [json.loads(l) for f in files for l in f.read_text().splitlines() if l.strip()]
    rows = run_cases(r, cases)
    failed = [x for x in rows if x["problems"]]
    print(f"cases: {len(rows)}   failed: {len(failed)}")
    for x in failed:
        print(f"  {x['id']}: {'; '.join(x['problems'])}")
    print("RESULT:", "PASS" if not failed else "FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
