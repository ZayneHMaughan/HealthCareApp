"""Plain-language "call emergency services if..." lines per symptom group.

Written to mirror the escalation logic in rules.py. They are authored by us (not cited sources), are placed in
the LLM prompt, and are what `when_to_seek_care` is allowed to draw on. DRAFT: needs clinician review, and the
988 reference is US-specific (see resources.py).
"""
from __future__ import annotations
from .schema import Intake

DEFAULT = "trouble breathing, chest pain, confusion, fainting, or symptoms that get worse quickly"

SIGNS = {
    "chest_pain": "pain spreading to your arm, jaw, or back; sweating; trouble breathing; or fainting",
    "shortness_of_breath": "breathlessness even at rest, being unable to speak a full sentence, or an oxygen reading below 92%",
    "headache": "a sudden severe headache, fever with a stiff neck, confusion, weakness or numbness, or trouble speaking",
    "new_onset_headache": "a sudden or worst-ever headache, fever with a stiff neck, confusion, weakness or numbness, or trouble speaking",
    "fever_chills": "a stiff neck, confusion, trouble breathing, a temperature of 103 F (39.5 C) or higher, or a fever lasting more than 3 days",
    "abdominal_pain": "severe pain, a hard or very tender belly, vomiting with fever, blood in vomit or stool, or fainting",
    "nausea_vomiting": "being unable to keep fluids down, blood in vomit, severe belly pain, or confusion",
    "bowel_changes": "blood in your stool, being unable to keep fluids down, or severe belly pain",
    "gi_bleeding": "heavy bleeding, dizziness, or fainting",
    "fainting": "chest pain, a racing or irregular heartbeat, or trouble breathing",
    "dizziness": "weakness or numbness on one side, trouble speaking, confusion, chest pain, or fainting",
    "palpitations": "chest pain, fainting, or trouble breathing",
    "back_pain": "weakness in a leg, numbness around the groin, or losing control of your bladder or bowels",
    "neck_pain": "fever with a very stiff neck, or numbness or weakness",
    "urinary_symptoms": "being unable to pass urine, blood in your urine, fever, or pain in your side",
    "head_face_injury": "passing out, repeated vomiting, confusion, or a worsening headache; if you take blood thinners, get checked even if you feel fine",
    "allergic_reaction": "swelling of the lips, face, or throat, a tight throat, or trouble breathing",
    "mouth_face_swelling": "trouble breathing or swallowing",
    "swelling_edema": "swelling in only one leg, shortness of breath, or chest pain",
    "vaginal_bleeding": "heavy bleeding, pregnancy with pain or bleeding, or dizziness",
    "pelvic_groin_pain": "pregnancy, fever, or sudden severe pain",
    "testicular_pain": "sudden, severe testicle pain, which needs emergency care",
    "seizure": "a first seizure, a seizure that does not stop, or trouble breathing afterwards",
    "burns": "burns larger than your palm, burns on the face, or burns from chemicals or electricity",
    "cuts_lacerations": "bleeding that does not slow after 10 minutes of firm pressure",
    "bleeding_bruising": "bleeding that does not slow after 10 minutes of firm pressure",
    "blood_sugar_problem": "vomiting, confusion, or unusual drowsiness",
    "dehydration": "being unable to keep fluids down, or confusion",
    "coughing_blood": "trouble breathing; coughing up blood is itself a reason to be seen today",
    "cough": "trouble breathing, coughing up blood, or chest pain",
    "cold_flu_sore_throat": "trouble breathing, fever with a stiff neck, or a fever lasting more than 3 days",
    "vision_problems": "sudden vision loss or sudden changes in your vision",
    "low_mood": "thoughts of harming yourself or someone else (in the US, call or text 988)",
    "anxiety_panic": "chest pain, trouble breathing, or thoughts of harming yourself (in the US, call or text 988)",
    "agitation_psychosis": "thoughts of harming yourself or someone else (in the US, call or text 988)",
}


def warning_signs_for(intake: Intake, max_groups: int = 3) -> list[str]:
    """Distinct warning-sign phrases for the intake's symptoms (most specific first), or the generic line."""
    out: list[str] = []
    for g in sorted(intake.symptoms):
        s = SIGNS.get(g)
        if s and s not in out:
            out.append(s)
    return out[:max_groups] or [DEFAULT]