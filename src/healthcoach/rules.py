"""Hard-coded red-flag layer. Runs BEFORE any model and can only RAISE urgency.

These thresholds are a draft for clinician review, not clinical guidance. Rules are
deliberately conservative (over-triage is preferred to under-triage), and the
free-text patterns ignore negation on purpose ("no chest pain" still matches).
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from .schema import Intake, Tier

CRISIS_RE = re.compile(
    r"\b(kill myself|suicid\w*|end(ing)? my life|want to die|don'?t want to live|hurt myself|self[- ]harm)\b", re.I)
EMERGENCY_RE = re.compile(
    r"\b(can'?t breathe|cannot breathe|not breathing|crushing chest|face (is )?droop\w*|slurred speech|"
    r"coughing up blood|vomiting blood|unconscious|passed out|overdos\w*)\b", re.I)

MIN_ADULT_AGE = 18


@dataclass
class RuleResult:
    floor: Tier = Tier.SELF_CARE
    reasons: list[str] = field(default_factory=list)
    crisis: bool = False
    out_of_scope: bool = False

    def raise_to(self, tier: Tier, reason: str) -> None:
        self.floor = max(self.floor, tier)
        self.reasons.append(f"[{tier.name}] {reason}")

    @property
    def emergency(self) -> bool:
        return self.floor >= Tier.EMERGENCY


def evaluate(i: Intake) -> RuleResult:
    r = RuleResult()
    S, M, v = i.symptoms, i.modifiers, i.vitals
    sev = i.severity or 0

    def s(*names): return any(n in S for n in names)
    def m(*names): return any(n in M for n in names)

    # --- scope -----------------------------------------------------------
    if i.age < MIN_ADULT_AGE:
        r.out_of_scope = True
        r.raise_to(Tier.SEE_DOCTOR_SOON, "Under 18: outside the supported age range; refer to a pediatric clinician")

    # --- crisis and free-text safety net -----------------------------------
    if s("suicidal_homicidal") or CRISIS_RE.search(i.free_text or ""):
        r.crisis = True
        r.raise_to(Tier.EMERGENCY, "Self-harm / harm-to-others language")
    if EMERGENCY_RE.search(i.free_text or ""):
        r.raise_to(Tier.EMERGENCY, "Emergency phrase in free text")

    # --- groups handled only by rules ---------------------------------------
    if s("stroke_alert", "unresponsive_arrest", "poisoning_overdose", "alcohol_withdrawal"):
        r.raise_to(Tier.EMERGENCY, "Emergency-only symptom group")
    if s("assault_violence"):
        r.raise_to(Tier.URGENT, "Assault: offer safety resources and care")

    # --- chest / heart -------------------------------------------------------
    assoc_cardiac = (s("shortness_of_breath", "fainting", "nausea_vomiting")
                     or m("radiating_arm_jaw", "sweating") or sev >= 7)
    if s("chest_pain"):
        r.raise_to(Tier.EMERGENCY if assoc_cardiac else Tier.URGENT, "Chest pain")
    if s("facial_jaw_pain") and (s("shortness_of_breath") or m("sweating")):
        r.raise_to(Tier.EMERGENCY, "Jaw/face pain with breathlessness or sweating (possible cardiac)")
    if s("fainting"):
        r.raise_to(Tier.EMERGENCY if s("chest_pain", "palpitations", "shortness_of_breath") else Tier.URGENT, "Fainting")
    if s("palpitations") and s("fainting", "chest_pain"):
        r.raise_to(Tier.EMERGENCY, "Palpitations with fainting or chest pain")

    # --- breathing -------------------------------------------------------------
    if s("shortness_of_breath"):
        severe = m("at_rest", "cant_speak_sentences") or sev >= 8 or (v.spo2 is not None and v.spo2 < 92)
        r.raise_to(Tier.EMERGENCY if severe else Tier.URGENT, "Shortness of breath")
    if s("coughing_blood"):
        r.raise_to(Tier.EMERGENCY if s("shortness_of_breath") else Tier.URGENT, "Coughing blood")

    # --- neuro -------------------------------------------------------------------
    if s("focal_weakness_numbness", "confusion_altered_mental") or m("speech_difficulty", "facial_droop"):
        r.raise_to(Tier.EMERGENCY, "Possible stroke / altered mental status")
    if s("vision_problems") and m("sudden_onset"):
        r.raise_to(Tier.EMERGENCY, "Sudden vision change")
    if s("seizure"):
        r.raise_to(Tier.EMERGENCY if m("first_seizure") else Tier.URGENT, "Seizure")
    if s("new_onset_headache"):
        if m("sudden_onset", "worst_ever"):
            r.raise_to(Tier.EMERGENCY, "Sudden / worst-ever headache")
        elif i.age >= 50:
            r.raise_to(Tier.SEE_DOCTOR_SOON, "New headache after age 50")
    if m("stiff_neck") and s("fever_chills", "new_onset_headache", "headache"):
        r.raise_to(Tier.EMERGENCY, "Stiff neck with fever/headache")
    if s("back_pain") and m("leg_weakness", "bladder_bowel_loss"):
        r.raise_to(Tier.EMERGENCY, "Back pain with leg weakness or bladder/bowel loss")

    # --- bleeding / allergy ----------------------------------------------------------
    if s("gi_bleeding"):
        r.raise_to(Tier.EMERGENCY if (s("dizziness", "fainting") or m("heavy_bleeding")) else Tier.URGENT, "GI bleeding")
    if s("allergic_reaction"):
        if s("shortness_of_breath", "mouth_face_swelling") or m("throat_tightness"):
            r.raise_to(Tier.EMERGENCY, "Allergic reaction with airway/breathing involvement")
        else:
            r.raise_to(Tier.SEE_DOCTOR_SOON, "Allergic reaction")

    # --- abdomen / pelvis / GU ----------------------------------------------------------
    if s("abdominal_pain") and (m("rigid_abdomen") or sev >= 8 or (s("fever_chills") and s("nausea_vomiting"))):
        r.raise_to(Tier.URGENT, "Abdominal pain with severe features")
    if s("testicular_pain"):
        r.raise_to(Tier.EMERGENCY if m("sudden_onset") else Tier.URGENT, "Testicular pain")
    if s("urinary_symptoms") and m("unable_to_urinate"):
        r.raise_to(Tier.URGENT, "Unable to urinate")
    if i.pregnant and s("abdominal_pain", "pelvic_groin_pain", "vaginal_bleeding"):
        r.raise_to(Tier.EMERGENCY if (m("heavy_bleeding") or sev >= 7) else Tier.URGENT, "Pregnancy with pain/bleeding")
    if i.pregnant and s("vision_problems", "new_onset_headache"):
        r.raise_to(Tier.URGENT, "Pregnancy with headache/vision change (possible preeclampsia)")

    # --- injury / burns -------------------------------------------------------------------
    if s("head_face_injury"):
        if m("loss_of_consciousness", "repeated_vomiting") or s("confusion_altered_mental"):
            r.raise_to(Tier.EMERGENCY, "Head injury with red-flag features")
        elif i.on_anticoagulants:
            r.raise_to(Tier.URGENT, "Head injury while on blood thinners")
    if s("burns") and m("large_burn", "face_or_airway", "chemical_or_electrical"):
        r.raise_to(Tier.EMERGENCY, "Severe burn")
    if s("swelling_edema"):
        if s("shortness_of_breath", "chest_pain"):
            r.raise_to(Tier.EMERGENCY, "Swelling with breathlessness/chest pain (possible clot)")
        elif m("one_leg"):
            r.raise_to(Tier.URGENT, "One-sided leg swelling (possible clot)")

    # --- metabolic / infection -------------------------------------------------------------
    if s("blood_sugar_problem") and s("nausea_vomiting"):
        r.raise_to(Tier.URGENT, "Blood sugar problem with vomiting")
    if s("dehydration") and m("unable_keep_fluids"):
        r.raise_to(Tier.URGENT, "Dehydration, cannot keep fluids down")
    if s("fever_chills"):
        if i.immunocompromised:
            r.raise_to(Tier.URGENT, "Fever while immunocompromised")
        if i.age >= 65:
            r.raise_to(Tier.SEE_DOCTOR_SOON, "Fever at age 65+")
        if (i.duration_hours or 0) > 72:
            r.raise_to(Tier.SEE_DOCTOR_SOON, "Fever longer than 3 days")

    # --- vitals ----------------------------------------------------------------------------
    if v.spo2 is not None and v.spo2 < 92:
        r.raise_to(Tier.EMERGENCY, f"SpO2 {v.spo2}")
    if v.systolic_bp is not None and v.systolic_bp < 90:
        r.raise_to(Tier.EMERGENCY, f"Systolic BP {v.systolic_bp}")
    if v.resp_rate is not None and v.resp_rate >= 30:
        r.raise_to(Tier.EMERGENCY, f"Respiratory rate {v.resp_rate}")
    if v.heart_rate is not None and (v.heart_rate >= 130 or v.heart_rate <= 40):
        r.raise_to(Tier.URGENT, f"Heart rate {v.heart_rate}")
    if v.temp_c is not None and v.temp_c >= 39.5:
        r.raise_to(Tier.URGENT, f"Temperature {v.temp_c} C")

    # --- bleeding anywhere / nothing recognized -------------------------------------------------
    if m("heavy_bleeding"):
        r.raise_to(Tier.EMERGENCY, "Heavy or uncontrolled bleeding")
    if not S and r.floor == Tier.SELF_CARE:
        r.raise_to(Tier.SEE_DOCTOR_SOON, "No recognized symptoms: cannot assess, advise clinician contact")

    # --- generic severity ---------------------------------------------------------------------
    if sev >= 9:
        r.raise_to(Tier.URGENT, "Severity 9-10/10")

    return r
