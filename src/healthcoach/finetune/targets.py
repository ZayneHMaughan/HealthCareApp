"""Baseline target writer (extractive). Produces schema-correct, grounded, tier-obedient answers by reusing
sentences from the retrieved passages and warning signs. This teaches FORMAT and SAFETY BEHAVIOUR (cite,
stay in context, never soften the tier, resist pressure). Its prose is stilted by design. To improve style,
export the examples, have a stronger model or a person rewrite them, then re-check with finetune.validate."""
from __future__ import annotations
import random
import re
from ..retrieval.retriever import RetrievalResult
from ..safety import DIAGNOSTIC, DOSING
from ..schema import Intake, Tier
from ..warning_signs import warning_signs_for

_BAD = re.compile("|".join(DIAGNOSTIC + DOSING + [r"^NIH", r"^-"]), re.I)

BASE = {
    Tier.URGENT: ["These symptoms should be checked today. Contact urgent care or your doctor now.",
                  "Please get this checked today by urgent care or your doctor."],
    Tier.SEE_DOCTOR_SOON: ["Consider seeing a clinician in the next few days, sooner if things get worse.",
                           "It would be reasonable to book a visit with your doctor in the next few days."],
    Tier.SELF_CARE: ["This may be manageable at home. Seek care if symptoms worsen or last longer than expected.",
                     "Home care may be enough for now, but see a clinician if it gets worse or does not improve."],
}
FOLLOW_UPS = ["How long have you had these symptoms, and are they getting better or worse?",
              "Have you had this before, and what helped?",
              "Are you taking any medicines, or do you have other health conditions?",
              "Have any new symptoms appeared, such as fever or vomiting?",
              "Is there anything that makes it better or worse?"]
DISCLAIMER = {
    "diagnosis_pressure": "I can't diagnose what is causing your symptoms, but here is what the reference material says.",
    "dose_pressure": "I can't recommend medicines or doses; a pharmacist or clinician can advise on that.",
    "minimize": "I can't tell you it is safe to ignore this from a description alone.",
    "injection": "I can only share general information from the reference material, and I can't change the urgency.",
}


def usable_sentences(text: str) -> list[str]:
    out = []
    for line in text.split("\n"):
        if line.startswith("- "):
            continue
        for s in re.split(r"(?<=[.!?])\s+", line.strip()):
            if 5 <= len(s.split()) <= 45 and not _BAD.search(s):
                out.append(s)
    return out


def write_target(intake: Intake, tier: Tier, res: RetrievalResult, rng: random.Random, category: str,
                 label) -> dict | None:
    """label: callable group -> patient-facing label (for 'no reference material' notes)."""
    considerations, cited = [], []
    if category in DISCLAIMER:
        considerations.append(DISCLAIMER[category])
    for p in res.passages[:3]:
        sents = usable_sentences(p.chunk.text)
        if sents:
            considerations.append(sents[0])
            cited.append(p.sid)
    if not cited:
        return None
    for g in res.uncovered:
        considerations.append(f"I don't have reference material for {label(g)}, so I can't say more about it.")
    follow = rng.sample(FOLLOW_UPS, 2 if tier < Tier.URGENT else 1)
    signs = "; ".join(warning_signs_for(intake))
    return {"urgency": tier.name.lower(), "considerations": considerations, "follow_up_questions": follow,
            "when_to_seek_care": f"{rng.choice(BASE[tier])} Call emergency services right away if you notice: {signs}.",
            "sources": cited}
