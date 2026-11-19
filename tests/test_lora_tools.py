import ast
import json
import os
from pathlib import Path

from healthcoach.finetune.build_dataset import build
from healthcoach.finetune.eval_generation import evaluate
from healthcoach.finetune.train_lora import to_prompt_completion
from healthcoach.llm import default_llm, schema_for_prompt
from healthcoach.retrieval import Retriever
from test_retrieval import index

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src/healthcoach"
HEAVY = {"torch", "transformers", "trl", "peft", "datasets", "accelerate", "llama_cpp", "bitsandbytes"}


def data(tmp_path, n=200):
    build("unused", str(tmp_path), n=n, seed=5, allow_synthetic=True, retriever=Retriever(index()))
    ev = [json.loads(l) for l in (tmp_path / "eval.jsonl").read_text().splitlines()]
    metas = [m for m in map(json.loads, (tmp_path / "meta.jsonl").read_text().splitlines()) if m["split"] == "eval"]
    return ev, metas


def raises(fn, *a, exc=Exception):
    try:
        fn(*a)
    except exc:
        return True
    return False


def test_prompt_completion_conversion():
    ex = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                       {"role": "assistant", "content": "a"}]}
    pc = to_prompt_completion(ex)
    assert [m["role"] for m in pc["prompt"]] == ["system", "user"] and pc["completion"] == [ex["messages"][2]]
    assert raises(to_prompt_completion, {"messages": ex["messages"][:2]}, exc=ValueError)


def test_heavy_libraries_are_only_imported_inside_functions():
    for f in ("finetune/train_lora.py", "finetune/merge_export.py", "llm.py"):
        top = [n for n in ast.parse((SRC / f).read_text()).body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = {(n.module or "").split(".")[0] if isinstance(n, ast.ImportFrom) else a.name.split(".")[0]
                for n in top for a in (n.names if isinstance(n, ast.Import) else [n])}
        assert not (mods & HEAVY), (f, mods & HEAVY)


def test_schema_pins_tier_and_allowed_sources(tmp_path):
    ev, _ = data(tmp_path)
    user = ev[0]["messages"][1]["content"]
    s = schema_for_prompt(user)
    assert len(s["properties"]["urgency"]["enum"]) == 1 and f'tier="{s["properties"]["urgency"]["enum"][0]}"' in user
    assert "S1" in s["properties"]["sources"]["items"]["enum"]
    assert raises(schema_for_prompt, "no tier here", exc=ValueError)


def test_every_training_target_conforms_to_the_inference_schema(tmp_path):
    ev, _ = data(tmp_path)
    train = [json.loads(l) for l in (tmp_path / "train.jsonl").read_text().splitlines()]
    for ex in train + ev:
        s = schema_for_prompt(ex["messages"][1]["content"])
        out = json.loads(ex["messages"][2]["content"])
        p = s["properties"]
        assert set(out) == set(s["required"])
        assert out["urgency"] in p["urgency"]["enum"]
        assert 1 <= len(out["considerations"]) <= p["considerations"]["maxItems"]
        assert len(out["follow_up_questions"]) <= p["follow_up_questions"]["maxItems"]
        assert out["sources"] and set(out["sources"]) <= set(p["sources"]["items"]["enum"])


def _reference(ev):
    ref = {e["messages"][1]["content"]: e["messages"][2]["content"] for e in ev}
    return lambda prompt: ref[prompt]


def test_generation_eval_perfect_model_has_zero_errors(tmp_path):
    ev, metas = data(tmp_path)
    rep = evaluate(ev, _reference(ev), metas)
    assert rep["n"] == len(ev) and rep["error_rate"] == 0 and rep["by_group"]


def test_generation_eval_catches_softening_bad_json_and_crashes(tmp_path):
    ev, metas = data(tmp_path)
    ref = _reference(ev)
    def soften(p):
        o = json.loads(ref(p)); o["urgency"] = "self_care"; return json.dumps(o)
    rep = evaluate(ev, soften, metas)
    assert rep["error_kinds"].get("tier_violation", 0) > 0 and 0 < rep["error_rate"] < 1
    assert evaluate(ev, lambda p: "not json", metas)["invalid_json_rate"] == 1.0
    assert evaluate(ev, lambda p: "[]", metas)["invalid_json_rate"] == 1.0          # JSON but not an object: no crash
    def boom(p): raise TimeoutError("slow")
    assert evaluate(ev, boom, metas)["error_rate"] == 1.0


def test_default_llm_explains_missing_model_path():
    os.environ.pop("HEALTHCOACH_GGUF", None)
    assert raises(default_llm, "x", exc=RuntimeError)


def test_samples_file_for_manual_review(tmp_path):
    from healthcoach.finetune.eval_generation import write_samples
    ev, metas = data(tmp_path)
    ref = _reference(ev)
    def soften(p):
        o = json.loads(ref(p)); o["urgency"] = "self_care"; return json.dumps(o)
    samples = []
    evaluate(ev, soften, metas, samples=samples)
    n = write_samples(samples, str(tmp_path / "review.txt"), n_pass=3, cap_fail=5)
    text = (tmp_path / "review.txt").read_text()
    assert n <= 8 and "MODEL OUTPUT:" in text and "PROMPT:" in text and "ERRORS:" in text and '"urgency"' in text
