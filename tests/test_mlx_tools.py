import ast
import json
from pathlib import Path

from healthcoach.finetune.build_dataset import build
from healthcoach.finetune.export_mlx import export_mlx
from healthcoach.finetune.fuse_mlx import check_base_model
from healthcoach.finetune.train_mlx import build_command, check_model, lora_config_yaml
from healthcoach.retrieval import Retriever
from test_retrieval import index

SRC = Path(__file__).parent.parent / "src/healthcoach"


def words(msgs):                      # stand-in for a real tokenizer: 1 token per word
    return sum(len(m["content"].split()) for m in msgs)


def make_data(tmp_path, budget=None, n=160):
    r = Retriever(index(), **({"budget_tokens": budget} if budget else {}))
    return build("unused", str(tmp_path), n=n, seed=2, allow_synthetic=True, retriever=r)


def test_export_uses_mlx_layout_one_example_per_line(tmp_path):
    make_data(tmp_path / "d")
    stats = export_mlx(tmp_path / "d", tmp_path / "m", words, max_seq_length=10_000)
    assert (tmp_path / "m/train.jsonl").exists() and (tmp_path / "m/valid.jsonl").exists()     # eval -> valid
    assert not (tmp_path / "m/eval.jsonl").exists()
    lines = (tmp_path / "m/train.jsonl").read_text().splitlines()
    assert len(lines) == stats["n_train"] and all(json.loads(l)["messages"][-1]["role"] == "assistant" for l in lines)
    assert stats["dropped_train"] == 0


def test_over_length_examples_are_dropped_not_truncated_and_mass_dropping_is_an_error(tmp_path):
    make_data(tmp_path / "d")
    rows = [json.loads(l) for l in (tmp_path / "d/train.jsonl").read_text().splitlines()]
    cut = sorted(words(r["messages"]) for r in rows)[len(rows) * 95 // 100]      # ~5% over the limit
    stats = export_mlx(tmp_path / "d", tmp_path / "m", words, max_seq_length=cut)
    assert 0 < stats["dropped_train"] <= len(rows) * 0.1
    out = [json.loads(l) for l in (tmp_path / "m/train.jsonl").read_text().splitlines()]
    assert all(words(o["messages"]) <= cut for o in out)
    try:
        export_mlx(tmp_path / "d", tmp_path / "m2", words, max_seq_length=50)
    except ValueError as e:
        assert "budget-tokens" in str(e)
    else:
        raise AssertionError("should refuse when most examples do not fit")


def test_smaller_retrieval_budget_gives_shorter_sequences(tmp_path):
    big = make_data(tmp_path / "a", budget=1100)["max_prompt_tokens"]
    small = make_data(tmp_path / "b", budget=60)["max_prompt_tokens"]
    assert small < big
    assert json.loads((tmp_path / "b/stats.json").read_text())["retrieval_budget_tokens"] == 60


def test_command_has_the_memory_and_masking_flags():
    cmd = build_command(["mlx_lm.lora"], model="m", data="d", adapters="a", config="c.yaml", iters=500, batch=1,
                        lr=2e-5, num_layers=8, max_seq=2048, save_every=50, eval_every=50, seed=7)
    for flag in ("--mask-prompt", "--grad-checkpoint", "--train"):
        assert flag in cmd
    assert cmd[cmd.index("--batch-size") + 1] == "1" and cmd[cmd.index("--iters") + 1] == "500"
    assert cmd[cmd.index("--config") + 1] == "c.yaml"


def test_lora_yaml():
    y = lora_config_yaml(16, 20.0, 0.05)
    assert "rank: 16" in y and "scale: 20.0" in y and y.startswith("lora_parameters:")


def test_mac_scripts_import_no_heavy_libraries_at_module_level():
    for f in ("finetune/export_mlx.py", "finetune/train_mlx.py", "finetune/fuse_mlx.py"):
        top = [n for n in ast.parse((SRC / f).read_text()).body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = {(n.module or "").split(".")[0] if isinstance(n, ast.ImportFrom) else a.name.split(".")[0]
                for n in top for a in (n.names if isinstance(n, ast.Import) else [n])}
        assert not (mods & {"mlx", "mlx_lm", "torch", "transformers"}), (f, mods)


def test_non_mlx_model_formats_are_rejected_before_training(tmp_path):
    for bad in ("unsloth/Llama-3.2-3B-Instruct-bnb-4bit", "TheBloke/Foo-GPTQ", "org/Model-AWQ", "org/model-GGUF"):
        try:
            check_model(bad)
        except SystemExit as e:
            assert "mlx-community" in str(e)
        else:
            raise AssertionError(f"{bad} should have been rejected")
    assert check_model("mlx-community/Llama-3.2-1B-Instruct-4bit") == []
    assert check_model("mlx-community/Llama-3.2-3B-Instruct-4bit")            # allowed, but warns about 8 GB
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "bitsandbytes"}}))
    try:
        check_model(str(tmp_path))
    except SystemExit:
        pass
    else:
        raise AssertionError("local CUDA-quantized checkpoint should be rejected")
    (tmp_path / "config.json").write_text(json.dumps({"quantization": {"bits": 4, "group_size": 64}}))
    assert check_model(str(tmp_path)) == []


def test_fuse_base_must_be_the_original_full_precision_model(tmp_path):
    for bad in ("mlx-community/Llama-3.2-1B-Instruct-4bit", "unsloth/Llama-3.2-3B-Instruct-bnb-4bit", "x/y-8bit"):
        try:
            check_base_model(bad)
        except SystemExit as e:
            assert "ORIGINAL" in str(e)
        else:
            raise AssertionError(bad)
    check_base_model("meta-llama/Llama-3.2-1B-Instruct")
    check_base_model("unsloth/Llama-3.2-1B-Instruct")
    (tmp_path / "config.json").write_text(json.dumps({"quantization": {"bits": 4}}))
    try:
        check_base_model(str(tmp_path))
    except SystemExit:
        pass
    else:
        raise AssertionError("local MLX-quantized dir should be rejected")


def _fused(tmp_path, quantized=False, tokenizer=True, weights=True):
    d = tmp_path / "fused"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"quantization": {"bits": 4}} if quantized else {"model_type": "llama"}))
    if weights:
        (d / "model.safetensors").write_bytes(b"x")
    if tokenizer:
        (d / "tokenizer.json").write_text("{}")
    return d


def test_make_gguf_input_checks(tmp_path):
    from healthcoach.finetune.make_gguf import check_fused
    def fails(d, needle):
        try:
            check_fused(d)
        except SystemExit as e:
            assert needle in str(e), str(e)
        else:
            raise AssertionError("should have failed")
    check_fused(_fused(tmp_path))                                                    # a good folder passes
    fails(tmp_path / "nope", "fuse_mlx")
    for i, (kw, needle) in enumerate([({"quantized": True}, "ORIGINAL"), ({"tokenizer": False}, "tokenizer"),
                                      ({"weights": False}, "safetensors")]):
        sub = tmp_path / f"c{i}"; sub.mkdir()
        fails(_fused(sub, **kw), needle)


def test_make_gguf_finds_llama_cpp_and_builds_commands(tmp_path):
    from healthcoach.finetune.make_gguf import commands, find_llama_cpp
    root = tmp_path / "llama.cpp"
    (root / "build/bin").mkdir(parents=True)
    try:
        find_llama_cpp(tmp_path / "missing")
    except SystemExit as e:
        assert "git clone" in str(e)
    (root / "convert_hf_to_gguf.py").write_text("")
    try:
        find_llama_cpp(root)                                       # script present, binary not built
    except SystemExit as e:
        assert "cmake" in str(e) and "GGML_METAL" in str(e)
    (root / "build/bin/llama-quantize").write_text("")
    script, quant = find_llama_cpp(root)
    conv, qcmd, f16, q = commands(script, quant, tmp_path / "fused", tmp_path / "gguf", "tuned", "Q4_K_M", "python3")
    assert conv[:2] == ["python3", str(script)] and "--outtype" in conv and conv[conv.index("--outtype") + 1] == "f16"
    assert qcmd == [str(quant), str(f16), str(q), "Q4_K_M"] and q.name == "tuned-q4_k_m.gguf"


def test_make_gguf_check_only_needs_no_llama_cpp(tmp_path):
    import subprocess, sys
    from healthcoach.finetune.make_gguf import summarize_fused
    d = _fused(tmp_path)
    (d / "config.json").write_text(json.dumps({"model_type": "llama", "torch_dtype": "bfloat16", "num_hidden_layers": 16}))
    s = summarize_fused(d)
    assert s["model_type"] == "llama" and s["dtype"] == "bfloat16" and s["layers"] == 16 and s["tokenizer"] == ["tokenizer.json"]
    env = {**__import__("os").environ, "PYTHONPATH": str(SRC.parent)}
    ok = subprocess.run([sys.executable, "-m", "healthcoach.finetune.make_gguf", "--fused", str(d), "--check-only"],
                        capture_output=True, text=True, env=env)
    assert ok.returncode == 0 and "looks usable" in ok.stdout
    (d / "config.json").write_text(json.dumps({"quantization": {"bits": 4}}))
    bad = subprocess.run([sys.executable, "-m", "healthcoach.finetune.make_gguf", "--fused", str(d), "--check-only"],
                         capture_output=True, text=True, env=env)
    assert bad.returncode != 0 and "ORIGINAL" in bad.stderr


def test_tokenizer_config_patch_is_targeted_safe_and_idempotent(tmp_path):
    from healthcoach.finetune.make_gguf import fix_tokenizer_config
    d = _fused(tmp_path)
    cfg = d / "tokenizer_config.json"
    cfg.write_text(json.dumps({"tokenizer_class": "TokenizersBackend", "chat_template": "KEEP ME", "bos_token": "<s>"}))
    assert fix_tokenizer_config(d, apply=False) is True and "TokenizersBackend" in cfg.read_text()   # dry check changes nothing
    assert fix_tokenizer_config(d) is True
    new = json.loads(cfg.read_text())
    assert new["tokenizer_class"] == "PreTrainedTokenizerFast" and new["chat_template"] == "KEEP ME" and new["bos_token"] == "<s>"
    assert json.loads((d / "tokenizer_config.json.bak").read_text())["tokenizer_class"] == "TokenizersBackend"
    assert fix_tokenizer_config(d) is False                                                              # idempotent
    other = tmp_path / "o"; other.mkdir(); o = _fused(other)
    (o / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "LlamaTokenizer"}))
    assert fix_tokenizer_config(o) is False and json.loads((o / "tokenizer_config.json").read_text())["tokenizer_class"] == "LlamaTokenizer"
    nojson = tmp_path / "n"; nojson.mkdir(); n = _fused(nojson, tokenizer=False)
    (n / "tokenizer.model").write_text("x")
    (n / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "TokenizersBackend"}))
    assert fix_tokenizer_config(n) is False                                                              # needs tokenizer.json
