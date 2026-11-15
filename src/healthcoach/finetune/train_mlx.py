"""LoRA / QLoRA fine-tune on Apple Silicon with mlx-lm. Intended target: M-series Mac with 8 GB unified memory.

  pip install "mlx-lm[train]"
  python -m healthcoach.finetune.train_mlx --model mlx-community/<small 4-bit instruct model> --data data/ft --work models/mlx_run --smoke
  python -m healthcoach.finetune.train_mlx --model ...                                       --data data/ft --work models/mlx_run

A 4-bit (quantized) base makes mlx-lm train QLoRA automatically. Loss is on the assistant reply only (--mask-prompt),
with gradient checkpointing and batch size 1 to keep memory low. Not run in the authoring environment (no Mac):
use --dry-run to see the exact mlx-lm command, and --smoke (50 iterations) to measure speed and memory first.
Hyperparameters are mlx-lm defaults/doc examples, not tuned for this task: judge them with the generation eval.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from .export_mlx import export_mlx, tokenizer_counter


_NOT_MLX = re.compile(r"(?<![a-z0-9])(bnb|gptq|awq|gguf|exl2)(?![a-z0-9])")


def check_model(model: str) -> list[str]:
    """Refuse model formats mlx-lm cannot load; return soft warnings. mlx-lm needs MLX-format weights
    (for example the mlx-community/...-4bit repos), not CUDA quantizations like bitsandbytes (bnb), GPTQ, AWQ, or GGUF."""
    warnings: list[str] = []
    if os.path.isdir(model):
        cfg = Path(model) / "config.json"
        if cfg.exists():
            c = json.loads(cfg.read_text())
            if "quantization_config" in c and "quantization" not in c:
                raise SystemExit(f"{model} looks like a CUDA-quantized (not MLX) checkpoint. Use an MLX-format model.")
    else:
        m = _NOT_MLX.search(model.lower())
        if m:
            raise SystemExit(
                f"'{model}' is a {m.group(1).upper()}-quantized checkpoint (made for CUDA/PyTorch); mlx-lm cannot load it.\n"
                "Use an MLX-format model instead, e.g. mlx-community/Llama-3.2-1B-Instruct-4bit "
                "(browse huggingface.co/mlx-community), or pass --force to try anyway.")
        if re.search(r"(?<![0-9.])(3|4|7|8)b(?![a-z])", model.lower()):
            warnings.append("Models of 3B+ parameters will likely run out of memory training on 8 GB. Try ~1B first.")
    return warnings


def mlx_entry(sub: str) -> list[str]:
    """`mlx_lm.<sub>` console script if installed, else `python -m mlx_lm <sub>`."""
    exe = shutil.which(f"mlx_lm.{sub}")
    return [exe] if exe else [sys.executable, "-m", "mlx_lm", sub]


def lora_config_yaml(rank: int, scale: float, dropout: float) -> str:
    return f"lora_parameters:\n  rank: {rank}\n  scale: {scale}\n  dropout: {dropout}\n"


def build_command(entry: list[str], *, model: str, data: str, adapters: str, config: str, iters: int, batch: int,
                  lr: float, num_layers: int, max_seq: int, save_every: int, eval_every: int, seed: int) -> list[str]:
    return entry + ["--model", model, "--train", "--data", data, "--adapter-path", adapters, "--config", config,
                    "--batch-size", str(batch), "--iters", str(iters), "--learning-rate", str(lr),
                    "--num-layers", str(num_layers), "--max-seq-length", str(max_seq), "--mask-prompt",
                    "--grad-checkpoint", "--steps-per-report", "10", "--steps-per-eval", str(eval_every),
                    "--save-every", str(save_every), "--seed", str(seed)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="quantized MLX model (repo id or local path), e.g. a 4-bit ~1-1.5B instruct model")
    ap.add_argument("--data", required=True, help="build_dataset output dir")
    ap.add_argument("--work", required=True, help="run directory: mlx data, config, adapters")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-layers", type=int, default=8, help="LoRA on the last N layers (fewer = less memory)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--iters", type=int, help="override the computed iteration count")
    ap.add_argument("--smoke", action="store_true", help="50 iterations: measure speed and peak memory first")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="skip the model-format check")
    a = ap.parse_args()

    if a.force:
        notes = []
    else:
        notes = check_model(a.model)
    for n in notes:
        print("WARNING:", n)
    work = Path(a.work)
    stats = export_mlx(a.data, work / "data", tokenizer_counter(a.model), a.max_seq_length)
    print("data:", stats)
    iters = 50 if a.smoke else a.iters or math.ceil(a.epochs * stats["n_train"] / a.batch_size)
    every = max(25, min(200, iters // 10))
    (work / "lora.yaml").write_text(lora_config_yaml(a.rank, a.scale, a.dropout))
    cmd = build_command(mlx_entry("lora"), model=a.model, data=str(work / "data"), adapters=str(work / "adapters"),
                        config=str(work / "lora.yaml"), iters=iters, batch=a.batch_size, lr=a.lr,
                        num_layers=a.num_layers, max_seq=a.max_seq_length, save_every=every, eval_every=every,
                        seed=a.seed)
    print("command:", " ".join(cmd))
    if a.dry_run:
        return
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"mlx-lm exited with code {e.returncode}. Its own error message is printed ABOVE this line "
                         "(scroll up). Common causes: a non-MLX model (bnb/GPTQ/AWQ), out of memory (use a smaller "
                         "model, --num-layers 4, shorter data), or an mlx-lm version whose flags differ "
                         "(run `mlx_lm.lora --help`).")
    print(f"adapters saved in {work / 'adapters'}.\nNext: python -m healthcoach.finetune.fuse_mlx --base-model <UNQUANTIZED HF id> "
          f"--adapters {work / 'adapters'} --out models/fused_hf")


if __name__ == "__main__":
    main()