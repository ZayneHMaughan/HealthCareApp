"""Validate any chat-format JSONL (ours, or targets rewritten by a person or a stronger model).

  python -m healthcoach.finetune.validate data/ft/train.jsonl

Errors (exit 1): invalid JSON, missing keys, urgency != the tier in the prompt, uncited or unknown sources,
diagnostic or dosing language, reassurance at urgent tiers.
Warnings: considerations with low lexical overlap with the passages they cite (possible ungrounded claims).
"""
from __future__ import annotations
import json
import re
import sys
from ..retrieval.textproc import tokenize
from ..safety import post_check
from ..schema import Tier

_OWN = re.compile(r"^I (can't|can’t|can only|don't|don’t)\b")   # our fixed disclaimers are not claims about sources
MIN_OVERLAP = 0.5


def check_example(ex: dict) -> tuple[list[str], list[str]]:
    errs, warns = [], []
    user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
    try:
        out = json.loads(next(m["content"] for m in ex["messages"] if m["role"] == "assistant"))
    except Exception as e:
        return [f"assistant is not valid JSON: {e}"], []
    m = re.search(r'<assessment tier="(\w+)"', user)
    if not m:
        return ["prompt has no assessment tier"], []
    tier = Tier[m.group(1).upper()]
    passages = {sid: txt for sid, txt in re.findall(r"^\[(S\d+)\][^\n]*\n(.*?)(?=\n\n\[S\d+\]|\n</reference_material>)",
                                                      user, re.S | re.M)}
    errs += post_check(out, tier, set(passages))
    if isinstance(out, dict) and str(out.get("urgency", "")).upper() != tier.name:
        errs.append(f"urgency {out.get('urgency')!r} != tier {tier.name.lower()!r} in prompt")
    cited = " ".join(passages.get(s, "") for s in (out.get("sources") or []) if isinstance(s, str))
    ref = set(tokenize(cited))
    for c in out.get("considerations", []) if isinstance(out, dict) else []:
        toks = tokenize(c)
        if toks and not _OWN.match(c) and len(set(toks) & ref) / len(set(toks)) < MIN_OVERLAP:
            warns.append(f"weakly grounded: {c[:80]!r}")
    return errs, warns


def main() -> int:
    bad = warn = n = 0
    for path in sys.argv[1:]:
        for ln, line in enumerate(open(path, encoding="utf-8"), 1):
            if not line.strip():
                continue
            n += 1
            errs, warns = check_example(json.loads(line))
            warn += bool(warns)
            if errs:
                bad += 1
                print(f"{path}:{ln}: {'; '.join(errs)}")
    print(f"{n} examples, {bad} with errors, {warn} with grounding warnings")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
