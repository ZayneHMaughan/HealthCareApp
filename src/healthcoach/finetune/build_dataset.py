"""Build supervised fine-tuning data for the explanation LLM.

  python -m healthcoach.finetune.build_dataset --index models/kb --out data/ft --n 3000 --seed 7

Outputs (out/): train.jsonl, eval.jsonl (chat format), meta.jsonl (one row per example), stats.json (data card).
Every example is produced by the real pipeline components (rules, retriever, prompt builder) and every target
passes safety.post_check. Train and eval are split by scenario signature so near-duplicates never straddle them.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from ..intake.flow import Flow
from ..pipeline import SYSTEM_PROMPT, _prompt
from ..retrieval import Index, Retriever
from ..retrieval.embed import build_embedder
from ..retrieval.retriever import coverage_report
from ..retrieval.textproc import est_tokens
from ..rules import evaluate
from ..safety import post_check
from ..schema import Tier
from .sampler import sample_intake, to_case_dict
from .targets import write_target

DEFAULT_QUOTAS = {Tier.SELF_CARE: 0.30, Tier.SEE_DOCTOR_SOON: 0.40, Tier.URGENT: 0.30}


def _signature(intake, category: str) -> str:
    key = (tuple(sorted(intake.symptoms)), tuple(sorted(intake.modifiers)), int(intake.age // 10), intake.sex, category)
    return hashlib.sha1(repr(key).encode()).hexdigest()


def build(index_dir: str, out_dir: str, n: int = 3000, seed: int = 7, embedder: str = "none",
          allow_synthetic: bool = False, max_tokens: int = 2400, p_mod: float = 0.15,
          quotas: dict | None = None, retriever: Retriever | None = None) -> dict:
    rng = random.Random(seed)
    if retriever is None:
        retriever = Retriever(Index.load(index_dir, build_embedder(embedder)))
    idx = retriever.index
    if not allow_synthetic and any("SYNTHETIC TEST TEXT" in c.text for c in idx.chunks):
        raise SystemExit("Index contains synthetic test fixtures. Build the index from real sources, "
                         "or pass --allow-synthetic for a mechanics-only test run.")
    flow = Flow()
    cov = coverage_report(retriever)
    covered = {g for g, t in cov.items() if t}
    if not covered:
        raise SystemExit("No symptom group has any retrievable passage; nothing to build from.")
    weights = {g: (3.0 if g in covered else 1.0) for g in retriever.labels}
    quotas = quotas or DEFAULT_QUOTAS
    target_n = {t: round(n * q) for t, q in quotas.items()}
    counts, rows, seen = Counter(), [], set()
    label = lambda g: retriever.labels[g]["label"].lower() if g in retriever.labels else g.replace("_", " ")
    stats = Counter()

    for _ in range(n * 60):
        if all(counts[t] >= target_n[t] for t in target_n):
            break
        intake, cat = sample_intake(rng, flow, weights, p_mod)
        r = evaluate(intake)
        if r.floor >= Tier.EMERGENCY or r.out_of_scope or r.crisis:
            stats["skipped_emergency_or_scope"] += 1
            continue
        tier, src = r.floor, "rules"
        if rng.random() < 0.35:                                  # simulate the classifier raising the tier
            sim = Tier(rng.randint(int(r.floor), int(Tier.URGENT)))
            if sim > tier:
                tier, src = sim, "classifier"
        if counts[tier] >= target_n.get(tier, 0):
            stats["skipped_quota"] += 1
            continue
        res = retriever(intake, tier)
        if not res.passages:
            stats["skipped_no_grounding"] += 1
            continue
        target = write_target(intake, tier, res, rng, cat, label)
        if target is None:
            stats["skipped_no_usable_sentence"] += 1
            continue
        bad = post_check(target, tier, set(res.ids))
        assert not bad, f"target failed safety check (bug in writer): {bad}"
        prompt = _prompt(intake, tier, res, r.reasons)
        tokens = est_tokens(SYSTEM_PROMPT) + est_tokens(prompt) + est_tokens(json.dumps(target))
        if tokens > max_tokens:
            stats["skipped_too_long"] += 1
            continue
        h = hashlib.sha1(prompt.encode()).hexdigest()
        if h in seen:
            stats["skipped_duplicate"] += 1
            continue
        seen.add(h)
        counts[tier] += 1
        rows.append({"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": prompt},
                                  {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}],
                     "meta": {"id": h[:12], "tier": tier.name.lower(), "tier_source": src, "category": cat,
                              "symptoms": sorted(intake.symptoms), "n_passages": len(res.passages),
                              "uncovered": res.uncovered, "prompt_tokens": tokens, "intake": to_case_dict(intake),
                              "split": "eval" if int(_signature(intake, cat), 16) % 10 == 0 else "train"}})

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "eval"):
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                if r["meta"]["split"] == split:
                    f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
    with open(out / "meta.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r["meta"], ensure_ascii=False) + "\n")
    info = {
        "n_examples": len(rows), "seed": seed, "targets": "extractive baseline (finetune/targets.py)",
        "by_split": dict(Counter(r["meta"]["split"] for r in rows)),
        "by_tier": dict(Counter(r["meta"]["tier"] for r in rows)),
        "by_tier_source": dict(Counter(r["meta"]["tier_source"] for r in rows)),
        "by_category": dict(Counter(r["meta"]["category"] for r in rows)),
        "by_symptom": dict(Counter(s for r in rows for s in r["meta"]["symptoms"])),
        "with_uncovered_symptom": sum(1 for r in rows if r["meta"]["uncovered"]),
        "symptom_groups_with_corpus_support": sorted(covered),
        "symptom_groups_without_support": sorted(set(retriever.labels) - covered),
        "max_prompt_tokens": max((r["meta"]["prompt_tokens"] for r in rows), default=0),
        "skipped": dict(stats), "index_manifest": idx.manifest,
        "provenance": "Synthetic intakes sampled from the intake schema; no ER-dataset rows used.",
    }
    (out / "stats.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--embedder", default="none")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--p-mod", type=float, default=0.15, help="chance each applicable red-flag modifier is set")
    ap.add_argument("--allow-synthetic", action="store_true")
    a = ap.parse_args()
    info = build(a.index, a.out, a.n, a.seed, a.embedder, a.allow_synthetic, p_mod=a.p_mod)
    print(f"{info['n_examples']} examples  split={info['by_split']}  tiers={info['by_tier']}")
    print(f"categories={info['by_category']}")
    print(f"symptom groups without corpus support ({len(info['symptom_groups_without_support'])}): "
          f"{info['symptom_groups_without_support']}")
    print(f"skipped={info['skipped']}")


if __name__ == "__main__":
    main()
