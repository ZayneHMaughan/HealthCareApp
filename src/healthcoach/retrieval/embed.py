"""Embedders. The index records an embedder's `name`; loading with a different one is refused."""
from __future__ import annotations
import zlib
import numpy as np
from .textproc import tokenize


class Embedder:
    name = "base"
    dim = 0

    def embed(self, texts: list[str], kind: str = "doc") -> np.ndarray:  # kind: "doc" | "query"
        raise NotImplementedError


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(n == 0, 1, n)).astype(np.float32)


class HashingEmbedder(Embedder):
    """Deterministic bag-of-words hashing vectors. NOT semantic: it exists so the dense path can be
    tested offline and as a no-download fallback. Use a real model for production."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def embed(self, texts, kind="doc"):
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            for tok in tokenize(t):
                h = zlib.crc32(tok.encode())
                m[r, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        return _normalize(m)


class SentenceTransformerEmbedder(Embedder):
    """pip install sentence-transformers. For BGE/E5-style models, set the query/doc prefixes they expect."""

    def __init__(self, model_name: str, query_prefix: str = "", doc_prefix: str = ""):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.qp, self.dp = query_prefix, doc_prefix
        self.name = f"st:{model_name}"
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts, kind="doc"):
        pre = self.qp if kind == "query" else self.dp
        return _normalize(np.asarray(self.model.encode([pre + t for t in texts], show_progress_bar=False)))


class LlamaCppEmbedder(Embedder):
    """Embeddings from a GGUF embedding model via llama-cpp-python (keeps one runtime for everything)."""

    def __init__(self, gguf_path: str, query_prefix: str = "", doc_prefix: str = ""):
        from llama_cpp import Llama
        self.llm = Llama(model_path=gguf_path, embedding=True, n_ctx=512, verbose=False)
        self.qp, self.dp = query_prefix, doc_prefix
        self.name = f"llama:{gguf_path.rsplit('/', 1)[-1]}"
        self.dim = len(self.llm.create_embedding("x")["data"][0]["embedding"])

    def embed(self, texts, kind="doc"):
        pre = self.qp if kind == "query" else self.dp
        rows = [self.llm.create_embedding(pre + t)["data"][0]["embedding"] for t in texts]
        return _normalize(np.asarray(rows, dtype=np.float32))


def build_embedder(spec: str | None) -> Embedder | None:
    """'none' | 'hashing' | 'st:<model name>' | 'llama:<path to .gguf>'"""
    if not spec or spec == "none":
        return None
    if spec == "hashing":
        return HashingEmbedder()
    if spec.startswith("st:"):
        return SentenceTransformerEmbedder(spec[3:])
    if spec.startswith("llama:"):
        return LlamaCppEmbedder(spec[6:])
    raise ValueError(f"unknown embedder spec: {spec}")
