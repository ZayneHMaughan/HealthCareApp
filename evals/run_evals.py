"""Evaluation harness.

  python evals/run_evals.py                          # rules only
  python evals/run_evals.py --model models/risk_model # rules + classifier
  python evals/run_evals.py --llm myllm:generate      # also test LLM output checks

Exit code is 1 if any case is under-triaged, a crisis/scope expectation is missed,
or the over-triage rate exceeds --max-overtriage. Under-triage is the failure that matters.
"""
from __future__ import annotations
import argparse, importlib, json, sys
from collections import defaultdict
from pathlib import Path

from healthcoach.pipeline import run
from healthcoach.schema import Intake, Tier


def load_cases(path: Path) -> list[dict]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    return [json.loads(l) for f in files for l in f.read_text().splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="evals/cases")
    ap.add_argument("--model")
    ap.add_argument("--llm", help="module:function taking a prompt and returning a JSON string")
    ap.add_argument("--max-overtriage", type=float, default=0.40)
    ap.add_argument("--report", default="evals/reports/latest.json")
    a = ap.parse_args()

    engine = None
    if a.model:
        from healthcoach.risk import RiskEngine
        engine = RiskEngine(a.model)
    llm = None
    if a.llm:
        mod, fn = a.llm.split(":")
        llm = getattr(importlib.import_module(mod), fn)

    cases, rows = load_cases(Path(a.cases)), []
    for c in cases:
        resp = run(Intake.from_dict(c["intake"]), engine=engine, llm=llm)
        lo = Tier[c["expect_min"].upper()]
        hi = Tier[c["expect_max"].upper()] if c.get("expect_max") else None
        problems = []
        if resp.tier < lo:
            problems.append(f"UNDER-TRIAGE: got {resp.tier.name}, need >= {lo.name}")
        if c.get("expect_crisis") and not resp.crisis:
            problems.append("MISSED CRISIS")
        if c.get("expect_out_of_scope") and not resp.out_of_scope:
            problems.append("MISSED OUT-OF-SCOPE")
        rows.append({"id": c["id"], "tags": c.get("tags", []), "tier": resp.tier.name,
                     "expect_min": lo.name, "expect_max": hi.name if hi else None,
                     "under": resp.tier < lo, "over": hi is not None and resp.tier > hi,
                     "has_max": hi is not None, "problems": problems,
                     "violations": resp.violations, "used_fallback": resp.used_fallback,
                     "reasons": resp.reasons})

    n_under = sum(r["under"] for r in rows)
    capped = [r for r in rows if r["has_max"]]
    n_over = sum(r["over"] for r in capped)
    over_rate = n_over / len(capped) if capped else 0.0

    by_tag = defaultdict(lambda: [0, 0])
    for r in rows:
        for t in r["tags"]:
            by_tag[t][0] += 1
            by_tag[t][1] += bool(r["problems"])

    print(f"cases: {len(rows)}   under-triaged: {n_under}   over-triaged: {n_over}/{len(capped)} ({over_rate:.0%})")
    print("\nby tag (cases / failures):")
    for t, (n, f) in sorted(by_tag.items()):
        print(f"  {t:24s} {n:3d} / {f}")
    failed = [r for r in rows if r["problems"]]
    if failed:
        print("\nFAILURES:")
        for r in failed:
            print(f"  {r['id']}: {'; '.join(r['problems'])}")
    if llm:
        v = [r for r in rows if r["violations"]]
        fb = sum(r["used_fallback"] for r in rows)
        print(f"\nLLM: {len(v)} cases with safety violations, {fb} fell back to template")
        for r in v:
            print(f"  {r['id']}: {r['violations']}")

    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(rows, indent=2))
    ok = not failed and over_rate <= a.max_overtriage
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
