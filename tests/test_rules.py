from healthcoach.pipeline import run
from healthcoach.rules import evaluate
from healthcoach.safety import post_check
from healthcoach.schema import Intake, Tier


def mk(**kw):
    return Intake.from_dict({"age": 40, **kw})


def test_tier_ordering():
    assert Tier.EMERGENCY > Tier.URGENT > Tier.SEE_DOCTOR_SOON > Tier.SELF_CARE


def test_crisis_in_free_text_overrides_everything():
    r = evaluate(mk(symptoms=["headache"], free_text="I want to kill myself"))
    assert r.crisis and r.emergency


def test_under_18_out_of_scope_but_emergency_rules_still_run():
    r = evaluate(mk(age=10, symptoms=["shortness_of_breath"], modifiers=["at_rest"]))
    assert r.out_of_scope and r.emergency


def test_rules_never_lower_tier():
    r = evaluate(mk(symptoms=["chest_pain"]))
    assert r.floor >= Tier.URGENT


def test_emergency_skips_llm():
    called = []
    resp = run(mk(symptoms=["stroke_alert"]), llm=lambda p: called.append(p) or "{}")
    assert resp.tier == Tier.EMERGENCY and not called


def test_post_check_blocks_softened_urgency_and_dosing():
    out = {"urgency": "self_care", "considerations": ["take 500 mg"], "follow_up_questions": [],
           "when_to_seek_care": "probably nothing"}
    v = post_check(out, Tier.URGENT)
    assert any("softened" in x for x in v) and any("dosing" in x for x in v)
