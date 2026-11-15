"""Convert build_dataset output to the folder layout mlx-lm expects (train.jsonl + valid.jsonl, chat format),
dropping examples that would not fit --max-seq-length (mlx-lm would truncate them, cutting the JSON answer).

  python -m healthcoach.finetune.export_mlx --data data/ft --out data/ft_mlx --model <mlx model id or path>
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def _load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def export_mlx(data_dir, out_dir, count_tokens, max_seq_length: int = 2048, max_drop_frac: float = 0.10) -> dict:
    """count_tokens: callable(messages) -> int using the TARGET model's chat template and tokenizer."""
    data, out = Path(data_dir), Path(out_dir)
    train, valid = _load(data / "train.jsonl"), _load(data / "eval.jsonl")
    lens = {id(e): count_tokens(e["messages"]) for e in train + valid}
    kept_t = [e for e in train if lens[id(e)] <= max_seq_length]
    kept_v = [e for e in valid if lens[id(e)] <= max_seq_length]
    dropped = len(train) - len(kept_t)
    if train and dropped / len(train) > max_drop_frac:
        raise ValueError(f"{dropped}/{len(train)} training examples exceed {max_seq_length} tokens. Rebuild the data "
                         f"with a smaller --budget-tokens (and use the same budget at inference), or raise the limit.")
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", kept_t), ("valid.jsonl", kept_v)):
        with open(out / name, "w", encoding="utf-8") as f:       # one example per line: mlx-lm reads line by line
            for e in rows:
                f.write(json.dumps({"messages": e["messages"]}, ensure_ascii=False) + "\n")
    return {"n_train": len(kept_t), "n_valid": len(kept_v), "dropped_train": dropped,
            "dropped_valid": len(valid) - len(kept_v), "max_tokens": max(lens.values(), default=0),
            "mean_tokens": sum(lens.values()) // max(1, len(lens))}


def tokenizer_counter(model: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    return lambda msgs: len(tok(tok.apply_chat_template(msgs, tokenize=False), add_special_tokens=False)["input_ids"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="the model whose tokenizer/chat template training will use")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    a = ap.parse_args()
    print(export_mlx(a.data, a.out, tokenizer_counter(a.model), a.max_seq_length))


if __name__ == "__main__":
    main()
