"""Generation-based evaluation: run a model over held-out prompts and grade what it actually writes.

  HEALTHCOACH_GGUF=models/gguf/base-q4_k_m.gguf  python -m healthcoach.finetune.eval_generation \\
      --data data/ft --llm healthcoach.llm:default_llm --report evals/reports/base.json
  HEALTHCOACH_GGUF=models/gguf/tuned-q4_k_m.gguf python -m healthcoach.finetune.eval_generation \\
      --data data/ft --llm healthcoach.llm:default_llm --report evals/reports/tuned.json --baseline evals/reports/base.json

Eval loss only says the model imitates the targets. This says whether its own outputs are valid JSON, keep
the tier, cite only given passages, avoid dosing/diagnosis, and stay grounded, overall and per category/tier.
"""
from __future__ import annotations
import argparse
import importlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from .validate import check_example

BUCKETS = [("not valid JSON", "invalid_json"), ("not an object", "invalid_json"), ("missing keys", "missing_keys"),
           ("urgency", "tier_violation"), ("sources", "citation_violation"), ("dosing", "dosing"),
           ("diagnostic", "diagnostic"), ("reassurance", "reassurance")]


def _bucket(err: str) -> str:
    return next((b for k, b in BUCKETS if k in err), "other")


def evaluate(examples: list[dict], llm, metas: list[dict] | None = None, limit: int | None = None) -> dict:
    n = bad = warned = 0
    kinds: Counter = Counter()
    by: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for i, ex in enumerate(examples[:limit]):
        system = next(m["content"] for m in ex["messages"] if m["role"] == "system")
        user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
        try:
            raw = llm(user)
        except Exception as e:
            raw = f"<<generation failed: {e}>>"
        errs, warns = check_example({"messages": [{"role": "system", "content": system},
                                                  {"role": "user", "content": user},
                                                  {"role": "assistant", "content": raw}]})
        n += 1
        bad += bool(errs)
        warned += bool(warns)
        for b in {_bucket(e) for e in errs}:
            kinds[b] += 1
        if metas:
            for key in (f"category:{metas[i]['category']}", f"tier:{metas[i]['tier']}"):
                by[key][0] += 1
                by[key][1] += bool(errs)
    return {"n": n, "error_rate": bad / n if n else 0.0, "grounding_warning_rate": warned / n if n else 0.0,
            "invalid_json_rate": kinds["invalid_json"] / n if n else 0.0,
            "error_kinds": dict(kinds), "by_group": {k: {"n": v[0], "error_rate": v[1] / v[0]} for k, v in sorted(by.items())}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--llm", required=True, help="module:function taking the user prompt, returning a JSON string")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--report")
    ap.add_argument("--baseline", help="earlier report to compare against")
    ap.add_argument("--max-error-rate", type=float, default=0.02)
    a = ap.parse_args()
    mod, fn = a.llm.split(":")
    llm = getattr(importlib.import_module(mod), fn)
    d = Path(a.data)
    examples = [json.loads(l) for l in (d / f"{a.split}.jsonl").read_text().splitlines() if l.strip()]
    metas = [m for m in map(json.loads, (d / "meta.jsonl").read_text().splitlines()) if m["split"] == a.split]
    rep = evaluate(examples, llm, metas, a.limit)
    print(f"n={rep['n']}  error_rate={rep['error_rate']:.1%}  invalid_json={rep['invalid_json_rate']:.1%}  "
          f"grounding_warnings={rep['grounding_warning_rate']:.1%}")
    print("error kinds:", rep["error_kinds"])
    for k, v in rep["by_group"].items():
        print(f"  {k:32s} n={v['n']:4d}  error_rate={v['error_rate']:.1%}")
    if a.baseline:
        b = json.loads(Path(a.baseline).read_text())
        for k in ("error_rate", "invalid_json_rate", "grounding_warning_rate"):
            print(f"  vs baseline {k}: {b[k]:.1%} -> {rep[k]:.1%}")
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(rep, indent=2))
    return 0 if rep["error_rate"] <= a.max_error_rate else 1


if __name__ == "__main__":
    sys.exit(main())
