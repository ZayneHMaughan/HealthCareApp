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
_CONDITIONAL = re.compile(r"\b(if|when|whether|unless|should) you\b", re.I)


def _statements(out: dict) -> str:
    """Declarative sentences from considerations and when_to_seek_care. Questions ("Do you have a fever?")
    and conditionals ("Call 911 if you have trouble breathing") are not diagnoses and are skipped."""
    parts = [s for s in (out.get("considerations") or []) if isinstance(s, str)]
    if isinstance(out.get("when_to_seek_care"), str):
        parts.append(out["when_to_seek_care"])
    sents = [s.strip() for p in parts for s in re.split(r"(?<=[.!?])\s+", p)]
    return " ".join(s for s in sents if s and not s.endswith("?") and not _CONDITIONAL.search(s)).lower()

def post_check(out: dict, tier: Tier, allowed_sources: set[str] | None = None) -> list[str]:
    """allowed_sources: ids of the passages the model was given. When provided, the output must cite
    at least one and may cite only ids it was given."""
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
    if allowed_sources is not None:
        cited = out.get("sources")
        if not isinstance(cited, list) or not cited:
            v.append("no sources cited")
        else:
            bad = [s for s in cited if s not in allowed_sources]
            if bad:
                v.append(f"cites sources it was not given: {bad}")
    text = json.dumps(out).lower()
    statements = _statements(out)
    for pat in DIAGNOSTIC:
        if re.search(pat, statements):
            v.append(f"diagnostic language: {pat}")
    for pat in DOSING:
        if re.search(pat, text):
            v.append(f"dosing language: {pat}")
    if tier >= Tier.URGENT:
        for pat in REASSURING:
            if re.search(pat, text):
                v.append(f"reassurance despite {tier.name}: {pat}")
    return v
