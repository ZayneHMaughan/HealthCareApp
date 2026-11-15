"""LoRA / QLoRA fine-tune of a small instruct model on the data from build_dataset.

  python -m healthcoach.finetune.train_lora --base-model <hf id or local path> --data data/ft --out models/lora

Loss is computed on the assistant's JSON reply only: examples are converted to conversational
prompt/completion format, which TRL trains on completion-only by default and which works with any chat
template (assistant_only_loss would need special {% generation %} markers in the template).

Written against the TRL docs (v0.25 / main). Not run in the authoring environment (no GPU). If an argument
name errors, check your installed versions' release notes (known drift: dtype vs torch_dtype,
max_length vs max_seq_length, eval_strategy vs evaluation_strategy).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path


def to_prompt_completion(ex: dict) -> dict:
    msgs = ex["messages"]
    if not msgs or msgs[-1]["role"] != "assistant":
        raise ValueError("last message must be the assistant reply")
    return {"prompt": msgs[:-1], "completion": [msgs[-1]]}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--data", required=True, help="directory with train.jsonl and eval.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--qlora", action="store_true", help="4-bit base (CUDA + bitsandbytes only)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--check-only", action="store_true", help="run the data/template preflight and stop")
    a = ap.parse_args()

    import torch
    import transformers, trl, peft
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    data = Path(a.data)
    train = [to_prompt_completion(e) for e in load_jsonl(data / "train.jsonl")]
    evals = [to_prompt_completion(e) for e in load_jsonl(data / "eval.jsonl")]
    tok = AutoTokenizer.from_pretrained(a.base_model)

    # ---- preflight: what exactly will the model see? ------------------------------------------------------------
    try:
        full = tok.apply_chat_template(train[0]["prompt"] + train[0]["completion"], tokenize=False)
        head = tok.apply_chat_template(train[0]["prompt"], tokenize=False, add_generation_prompt=True)
    except Exception as e:   # e.g. templates that reject the system role
        raise SystemExit(f"Chat template failed on a system+user+assistant example: {e}\n"
                         "Pick a model whose template supports a system role, or merge SYSTEM_PROMPT into the user "
                         "turn in BOTH training data and inference.")
    print("=== first training example, rendered with the chat template ===\n" + full + "\n=== end ===")
    if not full.startswith(head):
        print("WARNING: the rendered prompt is not a prefix of the full text for this template. The prompt/completion "
              "boundary used for loss masking may be off; inspect the rendering above before training.")
    lengths = [len(tok(tok.apply_chat_template(e["prompt"] + e["completion"], tokenize=False),
                       add_special_tokens=False)["input_ids"]) for e in train + evals]
    too_long = sum(n > a.max_length for n in lengths)
    print(f"{len(train)} train / {len(evals)} eval examples; tokens max={max(lengths)} mean={sum(lengths)//len(lengths)}; "
          f"over --max-length ({a.max_length}): {too_long}")
    if too_long:
        raise SystemExit("Some examples exceed --max-length; truncation would cut the JSON. Raise it or rebuild the data.")
    if a.check_only:
        return

    # ---- model + adapter config ----------------------------------------------------------------------------------
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    init_kwargs = {"dtype": dtype}
    if a.qlora:
        from transformers import BitsAndBytesConfig
        init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    lora = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, target_modules="all-linear",
                      task_type="CAUSAL_LM")
    steps_per_epoch = max(1, math.ceil(len(train) / (a.batch * a.grad_accum)))
    every = max(10, steps_per_epoch // 2)
    cfg = SFTConfig(
        output_dir=str(Path(a.out) / "checkpoints"), num_train_epochs=a.epochs, learning_rate=a.lr,
        lr_scheduler_type="cosine", warmup_ratio=0.03, per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum, max_length=a.max_length,
        bf16=use_bf16, fp16=not use_bf16, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=5, eval_strategy="steps",
        eval_steps=every, save_strategy="steps", save_steps=every, save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, report_to="none", seed=a.seed,
        model_init_kwargs=init_kwargs)        # completion-only loss is the default for prompt/completion data
    trainer = SFTTrainer(model=a.base_model, args=cfg, train_dataset=Dataset.from_list(train),
                         eval_dataset=Dataset.from_list(evals), processing_class=tok, peft_config=lora)
    trainer.model.print_trainable_parameters()
    trainer.train()

    out = Path(a.out)
    trainer.save_model(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))
    (out / "run_info.json").write_text(json.dumps({
        "base_model": a.base_model, "qlora": a.qlora, "args": vars(a), "n_train": len(train), "n_eval": len(evals),
        "train_sha256": hashlib.sha256((data / "train.jsonl").read_bytes()).hexdigest(),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__,
                     "peft": peft.__version__},
        "best_eval_loss": trainer.state.best_metric}, indent=2))
    print(f"adapter saved to {out / 'adapter'}")


if __name__ == "__main__":
    main()
