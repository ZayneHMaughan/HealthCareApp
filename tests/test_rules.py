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


def test_diagnostic_check_flags_assertions_but_not_questions_or_conditionals():
    from healthcoach.safety import post_check
    def out(**kw):
        base = {"urgency": "urgent", "considerations": ["Causes vary."], "follow_up_questions": [],
                "when_to_seek_care": "Get checked today.", "sources": ["S1"]}
        base.update(kw); return base
    ok = lambda o: not [v for v in post_check(o, Tier.URGENT, {"S1"}) if "diagnostic" in v]
    assert not ok(out(considerations=["You probably have the flu."]))
    assert not ok(out(when_to_seek_care="You definitely have an infection. Get checked."))
    assert ok(out(follow_up_questions=["Do you have a fever?"]))
    assert ok(out(when_to_seek_care="Call 911 if you have trouble breathing."))
    assert ok(out(considerations=["Tell us when you have new symptoms."]))


def test_diagnostic_check_reports_the_sentence_and_allows_obligations():
    from healthcoach.safety import post_check
    def viol(**kw):
        base = {"urgency": "urgent", "considerations": ["Causes vary."], "follow_up_questions": [],
                "when_to_seek_care": "Get checked today.", "sources": ["S1"]}
        base.update(kw)
        return [v for v in post_check(base, Tier.URGENT, {"S1"}) if "diagnostic" in v]
    v = viol(considerations=["Causes vary.", "You probably have the flu."])
    assert len(v) == 1 and "You probably have the flu." in v[0]            # the text is in the message
    assert not viol(when_to_seek_care="You have to get this checked today.")  # obligation, not a diagnosis
    assert viol(when_to_seek_care="You have pneumonia.")


def test_benign_second_person_phrasing_is_allowed_but_real_diagnoses_are_not():
    from healthcoach.safety import post_check
    def flagged(sentence):
        o = {"urgency": "urgent", "considerations": [sentence], "follow_up_questions": [],
             "when_to_seek_care": "Get checked today.", "sources": ["S1"]}
        return any("diagnostic" in v for v in post_check(o, Tier.URGENT, {"S1"}))
    # seen in real model output: not diagnoses
    assert not flagged("Treatment for back pain depends on what kind of pain you have, and what is causing it.")
    assert not flagged("Once you've had a heart attack, you have a higher risk of having another one.")
    assert not flagged("You have to get this checked today.")
    # must still be caught, including attempts to hide them behind an exempt phrase
    for bad in ("You have pneumonia.", "You probably have the flu.", "Because you have an infection, you should rest.",
                "You have a higher risk of this and you have pneumonia.", "What you have is pneumonia.",
                "What you need to know is that you have diabetes.", "You have to rest, and you have pneumonia."):
        assert flagged(bad), bad


def test_normalize_output_removes_repeats_and_tolerates_odd_shapes():
    from healthcoach.pipeline import run
    from healthcoach.safety import normalize_output
    o = normalize_output({"follow_up_questions": ["Q one?", "q one? ", "Q two?"], "considerations": ["A.", "A."], "sources": ["S1", "S1"]})
    assert o["follow_up_questions"] == ["Q one?", "Q two?"] and o["considerations"] == ["A."] and o["sources"] == ["S1"]
    assert normalize_output([1, 2]) == [1, 2] and normalize_output({"considerations": "not a list"})["considerations"] == "not a list"
