"""Fuse a MLX LoRA adapter into the UNQUANTIZED base model, then convert to GGUF with llama.cpp.

  python -m healthcoach.finetune.fuse_mlx --base-model <original HF id, e.g. the fp16 repo> --adapters models/mlx_run/adapters --out models/fused_hf

llama.cpp's converter needs Hugging Face-format fp16/bf16 weights, so fuse into the original (not the 4-bit) model.
The adapter was trained against the 4-bit base, a standard QLoRA setup, but always evaluate the final quantized GGUF.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from .train_mlx import mlx_entry


_QUANT = re.compile(r"(?<![a-z0-9])(\d+bit|bnb|gptq|awq|gguf|exl2)(?![a-z0-9])")


def check_base_model(base: str) -> None:
    """The base must be the ORIGINAL full-precision model, not a quantized copy (llama.cpp's converter needs
    fp16/bf16 weights, and fusing into 4-bit weights would just give another 4-bit model)."""
    if os.path.isdir(base):
        cfg = Path(base) / "config.json"
        if cfg.exists() and "quantization" in json.loads(cfg.read_text()):
            raise SystemExit(f"{base} is an MLX-quantized model. Point --base-model at the original full-precision model.")
        return
    m = _QUANT.search(base.lower())
    if m:
        raise SystemExit(f"'{base}' looks quantized ({m.group(1)}). --base-model must be the ORIGINAL full-precision model: "
                         "open the MLX model's page on huggingface.co and look for 'converted to MLX format from <id>' "
                         "in its README (or the 'Base model' link in the Model tree).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="skip the base-model check")
    a = ap.parse_args()
    if not a.force:
        check_base_model(a.base_model)
    cmd = mlx_entry("fuse") + ["--model", a.base_model, "--adapter-path", a.adapters, "--save-path", a.out]
    print("command:", " ".join(cmd))
    if not a.dry_run:
        subprocess.run(cmd, check=True)
    print(f"\nThen, from a llama.cpp checkout (build it with Metal; see its README):\n"
          f"  python convert_hf_to_gguf.py {a.out} --outtype f16 --outfile models/gguf/model-f16.gguf\n"
          f"  ./build/bin/llama-quantize models/gguf/model-f16.gguf models/gguf/model-q4_k_m.gguf Q4_K_M\n"
          "and evaluate the quantized file:  HEALTHCOACH_GGUF=models/gguf/model-q4_k_m.gguf python -m healthcoach.finetune.eval_generation ...")


if __name__ == "__main__":
    main()