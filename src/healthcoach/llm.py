"""Local LLM adapter (llama.cpp) with grammar-constrained JSON.

The schema is built per call from the prompt itself: `urgency` can only be the tier the pipeline assigned and
`sources` can only be passage ids that appear in the prompt. So softening the tier or citing an invented
source is impossible by construction; safety.post_check remains as a second line of defence.

  HEALTHCOACH_GGUF=models/gguf/model-q4_k_m.gguf python evals/run_evals.py --index models/kb --llm healthcoach.llm:default_llm
"""
from __future__ import annotations
import os
import platform
import re
import sys
from .pipeline import build_messages

_TIER = re.compile(r'<assessment tier="(\w+)"')
_SID = re.compile(r"^\[(S\d+)\]", re.M)


def schema_for_prompt(prompt: str) -> dict:
    m = _TIER.search(prompt)
    ids = sorted(set(_SID.findall(prompt)))
    if not m or not ids:
        raise ValueError("prompt has no assessment tier or no reference passages")
    return {
        "type": "object",
        "properties": {
            "urgency": {"type": "string", "enum": [m.group(1)]},
            "considerations": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 6},
            "follow_up_questions": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
            "when_to_seek_care": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "string", "enum": ids}, "minItems": 1, "maxItems": len(ids)},
        },
        "required": ["urgency", "considerations", "follow_up_questions", "when_to_seek_care", "sources"],
        "additionalProperties": False,
    }


class LlamaCppLLM:
    """Callable prompt -> JSON string. Requires llama-cpp-python (pip install healthcoach[llm])."""

    def __init__(self, gguf_path: str, n_ctx: int = 4096, n_threads: int | None = None, n_gpu_layers: int = 0,
                 temperature: float = 0.1, max_tokens: int = 700):
        from llama_cpp import Llama
        self.llm = Llama(model_path=gguf_path, n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=n_gpu_layers,
                         verbose=False)
        self.temperature, self.max_tokens = temperature, max_tokens

    def __call__(self, prompt: str) -> str:
        out = self.llm.create_chat_completion(
            messages=build_messages(prompt),
            response_format={"type": "json_object", "schema": schema_for_prompt(prompt)},
            temperature=self.temperature, max_tokens=self.max_tokens)
        return out["choices"][0]["message"]["content"]


_default: LlamaCppLLM | None = None

def default_gpu_layers() -> int:
    """-1 (all layers on the GPU via Metal) on Apple Silicon, else 0. Override with $HEALTHCOACH_GPU_LAYERS."""
    apple = sys.platform == "darwin" and platform.machine() == "arm64"
    return int(os.environ.get("HEALTHCOACH_GPU_LAYERS", "-1" if apple else "0"))

def default_llm(prompt: str) -> str:
    """For `run_evals.py --llm healthcoach.llm:default_llm`; model path comes from $HEALTHCOACH_GGUF."""
    global _default
    if _default is None:
        path = os.environ.get("HEALTHCOACH_GGUF")
        if not path:
            raise RuntimeError("Set HEALTHCOACH_GGUF to the path of a .gguf model file.")
        apple = sys.platform == "darwin" and platform.machine() == "arm64"
        layers = int(os.environ.get("HEALTHCOACH_GPU_LAYERS", "-1" if apple else "0"))   # -1 = all layers on Metal
        _default = LlamaCppLLM(path, n_gpu_layers=layers)
    return _default(prompt)
