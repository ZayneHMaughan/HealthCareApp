import csv, json, re
from pathlib import Path

from healthcoach.intake.flow import DATA, Flow, IntakeSession, validate
from healthcoach.pipeline import run
from healthcoach.rules import evaluate
from healthcoach.schema import Intake, Tier

FLOW = Flow()
RULES_SRC = (Path(__file__).parent.parent / "src/healthcoach/rules.py").read_text()
SEED = Path(__file__).parent.parent / "evals/cases/seed.jsonl"


def raises(fn, *a):
    try:
        fn(*a)
    except ValueError:
        return True
    return False


def rule_modifiers() -> set[str]:
    return {x for call in re.findall(r"\bm\(([^)]*)\)", RULES_SRC) for x in re.findall(r'"([a-z_]+)"', call)}


def collectible_modifiers() -> set[str]:
    out = {q["modifier"] for q in FLOW.q["modifier_questions"]}
    for item in FLOW.q["screener"]["items"]:
        out |= set(item["modifiers"])
    return out


def test_every_rule_modifier_can_be_collected():
    missing = rule_modifiers() - collectible_modifiers()
    assert not missing, f"rules read modifiers the intake flow never asks: {sorted(missing)}"


def test_no_dead_questions():
    dead = collectible_modifiers() - rule_modifiers()
    assert not dead, f"intake asks for modifiers no rule uses: {sorted(dead)}"


def test_symptom_labels_cover_the_mapping():
    rows = list(csv.DictReader(open(DATA / "cc_symptom_mapping.csv")))
    feat = {r["group"] for r in rows if r["treatment"] == "feature"}
    picker = set(FLOW.sym["groups"])
    screener = {s for i in FLOW.q["screener"]["items"] for s in i["symptoms"]}
    assert feat <= picker | screener
    used = {s for q in FLOW.q["modifier_questions"] + FLOW.q["context_questions"] + FLOW.q["vitals"]
            for s in q["symptoms_any"]}
    assert used <= picker, f"questions reference unknown symptom groups: {sorted(used - picker)}"


def drive(answers: dict, default="no"):
    """Answer steps from a dict keyed by step id, using `default` for yes/no/unsure steps not listed."""
    s = IntakeSession()
    guard = 0
    while (step := s.current) is not None and guard < 60:
        guard += 1
        if step.id in answers:
            s.answer(answers[step.id])
        elif step.kind == "yes_no_unsure":
            s.answer(default)
        elif step.id == "screener":
            s.answer(["none"])
        elif step.id == "vitals_pick":
            s.answer([])
        elif step.kind == "scale":
            s.answer(0)
        elif step.kind == "single_select":
            s.answer(step.options[0]["value"])
        elif step.kind == "text":
            s.answer("")
        else:
            raise AssertionError(f"unscripted step {step.id}")
    return s


BASE = {"age": 40, "sex": "male", "symptoms": ["cough"]}


def test_halting_screener_item_stops_immediately_and_is_emergency():
    s = drive({"screener": ["one_sided"]})
    r = s.result()
    assert r.complete and r.halted_reason == "screener: one_sided" and "age" not in s.answers
    assert run(r.intake).tier == Tier.EMERGENCY
    assert any("Age was not collected" in a for a in r.assumptions)


def test_self_harm_screener_triggers_crisis_path():
    r = drive({"screener": ["self_harm"]}).result()
    resp = run(r.intake)
    assert resp.crisis and "988" in resp.message


def test_non_halting_item_continues():
    s = drive({**BASE, "screener": ["unsafe"]})
    assert "assault_violence" in s.result().intake.symptoms and "age" in s.answers


def test_early_exit_once_rules_reach_emergency():
    s = drive({"age": 58, "sex": "male", "symptoms": ["chest_pain"], "severity": 8})
    assert "severity" in s.answers and "duration" not in s.answers   # stopped right after severity
    assert s.result().complete


def test_unsure_policy_is_conservative_where_it_matters_and_recorded():
    s = drive({"age": 45, "sex": "male", "symptoms": ["back_pain"], "mod:bladder_bowel_loss": "unsure"})
    r = s.result()
    assert "bladder_bowel_loss" in r.intake.modifiers
    assert any("weren't sure" in a for a in r.assumptions)
    assert evaluate(r.intake).emergency
    s2 = drive({"age": 45, "sex": "male", "symptoms": ["abdominal_pain"], "mod:rigid_abdomen": "unsure"})
    assert "rigid_abdomen" not in s2.result().intake.modifiers


def test_back_undoes_an_answer():
    s = IntakeSession()
    s.answer(["none"]); s.answer(30.0)
    s.back()
    assert s.current.id == "age"


def test_validation():
    first = FLOW.next_step({})[0]
    assert raises(validate, first, ["none", "chest"])             # "none" can't be combined
    age = FLOW.next_step({"screener": ["none"]})[0]
    assert raises(validate, age, 200) and raises(validate, age, True)
    sym = FLOW.next_step({"screener": ["none"], "age": 40.0, "sex": "male"})[0]
    assert sym.id == "symptoms"
    assert raises(validate, sym, ["cough", "fever_chills", "chest_pain", "dizziness", "fainting", "seizure"])


def test_fahrenheit_converted_and_implausible_readings_rejected():
    s = drive({"age": 40, "sex": "male", "symptoms": ["fever_chills"], "vitals_pick": ["temp_c"],
               "vital:temp_c": {"value": 101.5, "unit": "F"}})
    assert abs(s.result().intake.vitals.temp_c - 38.6) < 0.1
    spo2 = FLOW.next_step({"screener": ["none"], "age": 40.0, "sex": "male", "symptoms": ["cough"],
                           "severity": 2, "duration": "about_a_day", "vitals_pick": ["spo2"]})[0]
    assert spo2.id == "vital:spo2" and raises(validate, spo2, 150)


def test_no_symptoms_never_ends_as_self_care():
    s = drive({"age": 40, "sex": "female", "symptoms": ["none"], "free_text": "just feel off"})
    assert run(s.result().intake).tier >= Tier.SEE_DOCTOR_SOON


def test_question_count_per_single_symptom():
    worst = 0
    for g in FLOW.sym["groups"]:
        s = drive({"age": 40, "sex": "male", "symptoms": [g]})
        worst = max(worst, len(s.answers))
    print("max answers for any single symptom:", worst)
    assert worst <= 14


def answers_from_intake(c: dict) -> dict:
    """Turn a seed eval case into scripted answers, so we can check the flow can express it."""
    it = c["intake"]
    mods, syms = set(it.get("modifiers", [])), set(it.get("symptoms", []))
    a = {"screener": [i["id"] for i in FLOW.q["screener"]["items"]
                      if set(i["symptoms"]) <= syms and set(i["modifiers"]) <= mods
                      and (i["symptoms"] or i["modifiers"])] or ["none"],
         "age": float(it["age"]), "sex": it.get("sex", "unknown"),
         "pregnant": "yes" if it.get("pregnant") else "no",
         "symptoms": [s for s in syms if s in FLOW.sym["groups"]] or ["none"],
         "severity": it.get("severity") or 0,
         "free_text": it.get("free_text", "")}
    hrs = it.get("duration_hours") or 24
    a["duration"] = min(FLOW.durations, key=lambda k: abs(FLOW.durations[k]["hours"] - hrs))
    for m in FLOW.mod_q:
        a[f"mod:{m}"] = "yes" if m in mods else "no"
    for f in FLOW.ctx_q:
        a[f"ctx:{f}"] = "yes" if it.get(f) else "no"
    v = it.get("vitals", {})
    a["vitals_pick"] = [k for k in v if k in FLOW.vit]
    for k in a["vitals_pick"]:
        a[f"vital:{k}"] = {"value": v[k], "unit": "C"} if k == "temp_c" else v[k]
    return a


def test_every_seed_eval_case_is_reachable_through_the_flow():
    bad = []
    for line in SEED.read_text().splitlines():
        c = json.loads(line)
        s = drive(answers_from_intake(c))
        flow_ev, orig_ev = evaluate(s.result().intake), evaluate(Intake.from_dict(c["intake"]))
        if flow_ev.floor < orig_ev.floor or flow_ev.crisis != orig_ev.crisis:
            bad.append((c["id"], flow_ev.floor.name, orig_ev.floor.name))
    assert not bad, f"flow loses information for: {bad}"
