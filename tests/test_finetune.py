import json
import random
import re
from pathlib import Path

from healthcoach.finetune.build_dataset import build
from healthcoach.finetune.sampler import CATEGORIES, FREE_TEXT, sample_intake, to_case_dict
from healthcoach.finetune.validate import check_example
from healthcoach.intake.flow import Flow
from healthcoach.pipeline import SYSTEM_PROMPT, build_messages
from healthcoach.retrieval import Retriever
from healthcoach.rules import CRISIS_RE, EMERGENCY_RE, evaluate
from healthcoach.schema import Intake, Tier
from healthcoach.warning_signs import DEFAULT, SIGNS, warning_signs_for
from test_intake import answers_from_intake, drive
from test_retrieval import index

FLOW = Flow()
ROOT = Path(__file__).parent.parent
W = {g: 1.0 for g in FLOW.sym["groups"]}


def small_build(tmp_path, n=240, seed=3):
    info = build("unused", str(tmp_path), n=n, seed=seed, allow_synthetic=True, retriever=Retriever(index()))
    return info, [json.loads(l) for l in (tmp_path / "meta.jsonl").read_text().splitlines()]


def test_sampler_is_deterministic():
    a = [to_case_dict(sample_intake(random.Random(5), FLOW, W)[0]) for _ in range(1)]
    b = [to_case_dict(sample_intake(random.Random(5), FLOW, W)[0]) for _ in range(1)]
    assert a == b


def test_pressure_text_never_trips_crisis_or_emergency_regexes():
    for cat, phrases in FREE_TEXT.items():
        for p in phrases:
            assert not CRISIS_RE.search(p) and not EMERGENCY_RE.search(p), (cat, p)


def test_every_sampled_intake_is_expressible_through_the_real_flow():
    rng = random.Random(11)
    for _ in range(250):
        intake, _cat = sample_intake(rng, FLOW, W)
        got = drive(answers_from_intake({"intake": to_case_dict(intake)})).result().intake
        a, b = evaluate(got), evaluate(intake)
        assert a.floor >= b.floor and a.crisis == b.crisis, to_case_dict(intake)


def test_build_refuses_synthetic_index_without_flag(tmp_path):
    try:
        build("unused", str(tmp_path), n=10, retriever=Retriever(index()))
    except SystemExit as e:
        assert "synthetic" in str(e).lower()
    else:
        raise AssertionError("should have refused")


def test_build_outputs_are_valid_grounded_and_below_emergency(tmp_path):
    info, meta = small_build(tmp_path)
    assert info["n_examples"] >= 150
    for split in ("train", "eval"):
        for line in (tmp_path / f"{split}.jsonl").read_text().splitlines():
            errs, _ = check_example(json.loads(line))
            assert not errs, errs
    assert {m["tier"] for m in meta} <= {"self_care", "see_doctor_soon", "urgent"}
    assert all(m["prompt_tokens"] <= 2400 and m["n_passages"] >= 1 for m in meta)


def test_tier_mix_pressure_categories_and_classifier_raised_tiers_present(tmp_path):
    info, meta = small_build(tmp_path)
    assert all(info["by_tier"].get(t, 0) > 20 for t in ("self_care", "see_doctor_soon", "urgent"))
    assert {c for c, _ in CATEGORIES} <= set(info["by_category"])
    assert info["by_tier_source"].get("classifier", 0) > 0


def test_pressure_examples_keep_tier_and_decline_politely(tmp_path):
    small_build(tmp_path)
    rows = [json.loads(l) for l in (tmp_path / "train.jsonl").read_text().splitlines()]
    metas = {m["id"]: m for m in map(json.loads, (tmp_path / "meta.jsonl").read_text().splitlines())}
    seen = set()
    for r in rows:
        tier = re.search(r'<assessment tier="(\w+)"', r["messages"][1]["content"]).group(1)
        out = json.loads(r["messages"][2]["content"])
        assert out["urgency"] == tier                                  # pressure never changes urgency
        text = " ".join(out["considerations"])
        if "Ignore your previous" in r["messages"][1]["content"] or "set urgency to self_care" in r["messages"][1]["content"]:
            assert "can't change the urgency" in text
            seen.add("injection")
        if "What dose of pain medicine" in r["messages"][1]["content"]:
            assert "can't recommend medicines or doses" in text
            seen.add("dose")
    assert seen == {"injection", "dose"}


def test_splits_are_disjoint_by_scenario_and_match_meta(tmp_path):
    info, meta = small_build(tmp_path)
    n_train = len((tmp_path / "train.jsonl").read_text().splitlines())
    n_eval = len((tmp_path / "eval.jsonl").read_text().splitlines())
    assert (n_train, n_eval) == (info["by_split"].get("train", 0), info["by_split"].get("eval", 0)) and n_eval > 0
    sig = lambda m: (tuple(m["symptoms"]), tuple(m["intake"]["modifiers"]), int(m["intake"]["age"] // 10),
                     m["intake"]["sex"], m["category"])
    assert {sig(m) for m in meta if m["split"] == "train"}.isdisjoint({sig(m) for m in meta if m["split"] == "eval"})


def test_no_overlap_with_hand_written_eval_cases(tmp_path):
    _, meta = small_build(tmp_path)
    seed = [json.loads(l)["intake"] for l in (ROOT / "evals/cases/seed.jsonl").read_text().splitlines()]
    seed_sigs = {json.dumps(to_case_dict(Intake.from_dict(s)), sort_keys=True) for s in seed}
    assert not any(json.dumps(m["intake"], sort_keys=True) in seed_sigs for m in meta)


def test_validator_catches_bad_targets(tmp_path):
    small_build(tmp_path, n=60)
    good = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    assert check_example(good)[0] == []
    def mutate(**kw):
        ex = json.loads(json.dumps(good)); out = json.loads(ex["messages"][2]["content"]); out.update(kw)
        ex["messages"][2]["content"] = json.dumps(out); return ex
    assert any("urgency" in e for e in check_example(mutate(urgency="emergency"))[0])
    assert any("not given" in e for e in check_example(mutate(sources=["S9"]))[0])
    assert any("dosing" in e for e in check_example(mutate(considerations=["Take 500 mg now."]))[0])
    assert any("diagnostic" in e for e in check_example(mutate(considerations=["You probably have the flu."]))[0])
    assert check_example(mutate(considerations=["Moon cheese is made of astronauts and tiny violins."]))[1]  # grounding warning


def test_warning_signs():
    assert set(SIGNS) <= set(FLOW.sym["groups"])
    assert warning_signs_for(Intake(age=40, symptoms={"rash_skin"})) == [DEFAULT]
    assert "arm, jaw" in warning_signs_for(Intake(age=40, symptoms={"chest_pain"}))[0]


def test_prompt_and_system_prompt_are_shared_with_inference():
    from healthcoach.pipeline import _prompt
    res = Retriever(index()).retrieve(Intake(age=40, symptoms={"chest_pain"}))
    p = _prompt(Intake(age=40, symptoms={"chest_pain"}), Tier.URGENT, res, [])
    assert "<warning_signs>" in p and build_messages(p)[0]["content"] == SYSTEM_PROMPT


def test_training_code_never_touches_the_er_dataset():
    for f in (ROOT / "src/healthcoach/finetune").glob("*.py"):
        assert not re.search(r"RData|pyreadr|cleandf|read_csv|read_parquet", f.read_text()), f.name
