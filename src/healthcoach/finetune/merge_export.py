"""Merge a LoRA adapter into the base model (in bf16, NOT the 4-bit base) and print the GGUF steps.

  python -m healthcoach.finetune.merge_export --base-model <same id> --adapter models/lora/adapter --out models/merged
"""
from __future__ import annotations
import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(a.base_model, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    merged.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.adapter).save_pretrained(a.out)
    print(f"merged model written to {a.out}\n\nNext (from a llama.cpp checkout / build):\n"
          f"  python convert_hf_to_gguf.py {a.out} --outfile models/gguf/model-f16.gguf --outtype f16\n"
          f"  ./llama-quantize models/gguf/model-f16.gguf models/gguf/model-q4_k_m.gguf Q4_K_M\n"
          "Then re-run the evals on the QUANTIZED file you will ship.")


if __name__ == "__main__":
    main()
