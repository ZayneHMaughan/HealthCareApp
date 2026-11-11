"""Post-checks on LLM output. Returns a list of violations (empty = pass)."""
from __future__ import annotations
import json
import re
from .schema import Tier

REQUIRED_KEYS = {"urgency", "considerations", "follow_up_questions", "when_to_seek_care"}

DIAGNOSTIC = [r"\byou (definitely |probably |likely )?have\b", r"\byou('re| are) (diagnosed|suffering from)\b",
              r"\bthis is (definitely |probably )?(a|an) \w+ (infection|attack|disease)\b"]
DOSING = [r"\b\d+(\.\d+)?\s?(mg|mcg|ml|iu)\b", r"\btake \d+\b"]
REASSURING = [r"nothing to worry", r"probably nothing", r"no need to (see|go|seek)", r"not serious", r"nothing serious"]


def post_check(out: dict, tier: Tier) -> list[str]:
    v: list[str] = []
    missing = REQUIRED_KEYS - set(out)
    if missing:
        v.append(f"missing keys: {sorted(missing)}")
    try:
        said = Tier[str(out.get("urgency", "")).upper()]
        if said < tier:
            v.append(f"urgency softened: said {said.name}, required {tier.name}")
    except KeyError:
        v.append(f"invalid urgency value: {out.get('urgency')!r}")
    text = json.dumps(out).lower()
    for pat in DIAGNOSTIC:
        if re.search(pat, text):
            v.append(f"diagnostic language: {pat}")
    for pat in DOSING:
        if re.search(pat, text):
            v.append(f"dosing language: {pat}")
    if tier >= Tier.URGENT:
        for pat in REASSURING:
            if re.search(pat, text):
                v.append(f"reassurance despite {tier.name}: {pat}")
    return v
