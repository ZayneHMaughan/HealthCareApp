"""Fused Hugging Face-format model -> f16 GGUF -> quantized GGUF (llama.cpp), with sanity checks.

  python -m healthcoach.finetune.make_gguf --fused models/fused_hf --llama-cpp ~/llama.cpp --out models/gguf --name tuned

Prerequisites: a llama.cpp checkout built with cmake (see README), and the converter's Python requirements installed
in the interpreter given by --convert-python (use a separate venv so torch/transformers don't clash with mlx-lm).
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def check_fused(fused: Path) -> None:
    if not fused.is_dir():
        raise SystemExit(f"{fused} is not a directory. Run fuse_mlx first (its --out folder is what goes here).")
    cfg = fused / "config.json"
    if not cfg.exists():
        raise SystemExit(f"{fused} has no config.json: not a Hugging Face-format model folder.")
    if "quantization" in json.loads(cfg.read_text()):
        raise SystemExit(f"{fused} is MLX-quantized. Fuse the adapter into the ORIGINAL full-precision model "
                         "(fuse_mlx --base-model <original id>) and point --fused at that output.")
    if not list(fused.glob("*.safetensors")):
        raise SystemExit(f"{fused} contains no .safetensors weights.")
    if not any((fused / t).exists() for t in ("tokenizer.json", "tokenizer.model")):
        raise SystemExit(f"{fused} has no tokenizer files (tokenizer.json / tokenizer.model).")


def fix_tokenizer_config(fused: Path, apply: bool = True) -> bool:
    """Newer transformers versions (used by mlx-lm when it saves the fused model) write
    "tokenizer_class": "TokenizersBackend", which older transformers (in llama.cpp's converter environment) cannot
    load. When tokenizer.json exists, the generic PreTrainedTokenizerFast is equivalent. A .bak copy is kept.
    Returns True if the config needed (and, with apply=True, received) the change."""
    cfg = fused / "tokenizer_config.json"
    if not cfg.exists() or not (fused / "tokenizer.json").exists():
        return False
    data = json.loads(cfg.read_text())
    if data.get("tokenizer_class") != "TokenizersBackend":
        return False
    if apply:
        bak = cfg.with_name("tokenizer_config.json.bak")
        if not bak.exists():
            shutil.copy2(cfg, bak)
        data["tokenizer_class"] = "PreTrainedTokenizerFast"
        cfg.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return True


def summarize_fused(fused: Path) -> dict:
    cfg = json.loads((fused / "config.json").read_text())
    files = sorted(fused.glob("*.safetensors"))
    return {"weights_gb": round(sum(f.stat().st_size for f in files) / 1e9, 2), "weight_files": len(files),
            "model_type": cfg.get("model_type"), "dtype": cfg.get("torch_dtype") or cfg.get("dtype"),
            "layers": cfg.get("num_hidden_layers"), "tokenizer": [n for n in ("tokenizer.json", "tokenizer.model")
                                                                  if (fused / n).exists()]}


def find_llama_cpp(root: Path) -> tuple[Path, Path]:
    script = root / "convert_hf_to_gguf.py"
    if not script.exists():
        raise SystemExit(f"{script} not found. Clone it: git clone https://github.com/ggml-org/llama.cpp {root}")
    quant = root / "build" / "bin" / "llama-quantize"
    if not quant.exists():
        found = shutil.which("llama-quantize")
        if not found:
            raise SystemExit(f"{quant} not found. Build it:\n  cmake -S {root} -B {root}/build -DGGML_METAL=ON\n"
                             f"  cmake --build {root}/build --config Release -j 8")
        quant = Path(found)
    return script, quant


def commands(script: Path, quant_bin: Path, fused: Path, out: Path, name: str, quant: str, python: str):
    f16, q = out / f"{name}-f16.gguf", out / f"{name}-{quant.lower()}.gguf"
    convert = [python, str(script), str(fused), "--outtype", "f16", "--outfile", str(f16)]
    quantize = [str(quant_bin), str(f16), str(q), quant]
    return convert, quantize, f16, q


def _size(p: Path) -> str:
    return f"{p.stat().st_size / 1e9:.2f} GB" if p.exists() else "missing"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", required=True)
    ap.add_argument("--llama-cpp", help="path to your llama.cpp checkout (not needed with --check-only)")
    ap.add_argument("--out")
    ap.add_argument("--check-only", action="store_true", help="only verify the fused folder, then stop")
    ap.add_argument("--name", default="model")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--convert-python", default=sys.executable, help="interpreter that has llama.cpp's converter requirements")
    ap.add_argument("--no-fix-tokenizer", action="store_true", help="do not patch tokenizer_config.json")
    ap.add_argument("--keep-f16", action="store_true", help="keep the large intermediate f16 file")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    fused = Path(a.fused).expanduser()
    check_fused(fused)
    info = summarize_fused(fused)
    print(f"Fused model looks usable: {info}")
    needs_fix = fix_tokenizer_config(fused, apply=not (a.check_only or a.no_fix_tokenizer))
    if needs_fix and (a.check_only or a.no_fix_tokenizer):
        print("NOTE: tokenizer_config.json names tokenizer_class 'TokenizersBackend', which the converter's older "
              "transformers cannot load. Run without --check-only to patch it automatically (a .bak copy is kept).")
    elif needs_fix:
        print("Patched tokenizer_config.json: tokenizer_class TokenizersBackend -> PreTrainedTokenizerFast "
              "(original saved as tokenizer_config.json.bak).")
    if a.check_only:
        print("Expected size: roughly 2 bytes per parameter (about 2.5 GB for a 1B model). A much smaller folder "
              "means the adapter was fused into a 4-bit model.")
        return
    if not a.llama_cpp or not a.out:
        raise SystemExit("--llama-cpp and --out are required unless you use --check-only")
    root, out = Path(a.llama_cpp).expanduser(), Path(a.out)
    script, quant_bin = find_llama_cpp(root)
    convert, quantize, f16, q = commands(script, quant_bin, fused, out, a.name, a.quant, a.convert_python)
    print("1)", " ".join(convert), "\n2)", " ".join(quantize))
    if a.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(convert, check=True)
    subprocess.run(quantize, check=True)
    print(f"\nf16: {_size(f16)}   {a.quant}: {_size(q)}")
    if not a.keep_f16 and q.exists() and q.stat().st_size > 0:
        f16.unlink(missing_ok=True)
    print(f"\nSmoke test (5 held-out prompts):\n  HEALTHCOACH_GGUF={q} python -m healthcoach.finetune.eval_generation "
          f"--data data/ft --llm healthcoach.llm:default_llm --limit 5")


if __name__ == "__main__":
    main()