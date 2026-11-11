"""rules -> classifier -> (retrieval + LLM) -> safety check. Final tier = max of all layers."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Callable
from .resources import RESOURCES
from .rules import evaluate
from .safety import post_check
from .schema import Intake, Tier

LLM = Callable[[str], str]            # prompt -> JSON string
Retriever = Callable[[Intake], str]   # intake -> vetted passages


@dataclass
class Response:
    tier: Tier
    reasons: list[str]
    crisis: bool = False
    out_of_scope: bool = False
    risk_score: float | None = None
    message: str = ""
    llm_output: dict | None = None
    violations: list[str] = field(default_factory=list)
    used_fallback: bool = False


def _template(tier: Tier, crisis: bool, out_of_scope: bool, region: str) -> str:
    res = RESOURCES[region]
    if crisis:
        return res["crisis"]
    if tier == Tier.EMERGENCY:
        return res["emergency"]
    if out_of_scope:
        return "This tool is for adults. Please contact a pediatric clinician about these symptoms."
    return {
        Tier.URGENT: "These symptoms should be checked today. Contact urgent care or your doctor now.",
        Tier.SEE_DOCTOR_SOON: "Consider seeing a clinician in the next few days, sooner if things get worse.",
        Tier.SELF_CARE: "This may be manageable at home. Seek care if symptoms worsen or last longer than expected.",
    }[tier]


def _prompt(i: Intake, tier: Tier, passages: str) -> str:
    return (f"<intake>{json.dumps(i, default=lambda o: sorted(o) if isinstance(o, set) else o.__dict__)}</intake>\n"
            f"<risk_assessment>tier: {tier.name.lower()}</risk_assessment>\n"
            f"<reference_material>{passages}</reference_material>\n"
            "Return JSON with keys: urgency, considerations, follow_up_questions, when_to_seek_care. "
            "Use only the reference material. Do not diagnose. The tier is final: never soften it.")


def run(i: Intake, engine=None, llm: LLM | None = None, retriever: Retriever | None = None,
        region: str = "US") -> Response:
    rules = evaluate(i)
    tier, score = rules.floor, None
    if engine is not None and not rules.crisis:
        score, model_tier = engine.predict(i)
        tier = max(tier, model_tier)

    resp = Response(tier=tier, reasons=rules.reasons, crisis=rules.crisis,
                    out_of_scope=rules.out_of_scope, risk_score=score)
    fallback = _template(tier, rules.crisis, rules.out_of_scope, region)

    # Emergencies, crises, and out-of-scope users never wait on a language model.
    if llm is None or tier >= Tier.EMERGENCY or rules.out_of_scope:
        resp.message = fallback
        return resp

    try:
        out = json.loads(llm(_prompt(i, tier, retriever(i) if retriever else "")))
        resp.llm_output = out
        resp.violations = post_check(out, tier)
    except Exception as e:  # invalid JSON, timeout, etc.
        resp.violations = [f"llm failure: {e}"]
    if resp.violations:
        resp.used_fallback = True
        resp.message = fallback
    else:
        resp.message = json.dumps(resp.llm_output)
    return resp
