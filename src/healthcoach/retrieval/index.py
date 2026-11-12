"""On-disk hybrid index: BM25 (rebuilt at load, no pickles) + optional dense vectors, fused with RRF.

Layout:  chunks.jsonl   vectors.npy (optional)   manifest.json
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import numpy as np
from .bm25 import BM25
from .embed import Embedder
from .types import Chunk

RRF_K = 60
POOL = 100


@dataclass
class Hit:
    idx: int
    chunk: Chunk
    rrf: float
    bm25: float
    dense: float | None
    tag_match: bool


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray | None = None,
                 embedder: Embedder | None = None, manifest: dict | None = None):
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder if vectors is not None else None
        self.manifest = manifest or {}
        self.bm25 = BM25([c.index_text for c in chunks])
        self.by_topic: dict[str, set[int]] = {}
        for i, c in enumerate(chunks):
            for t in c.topics:
                self.by_topic.setdefault(t, set()).add(i)

    # ---- build / persist ---------------------------------------------------------------------------
    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder | None = None) -> "Index":
        vectors = embedder.embed([c.index_text for c in chunks], "doc") if embedder and chunks else None
        return cls(chunks, vectors, embedder)

    def save(self, path: str | Path) -> dict:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        blob = "".join(json.dumps(c.to_dict(), sort_keys=True, ensure_ascii=False) + "\n" for c in self.chunks)
        (p / "chunks.jsonl").write_text(blob, encoding="utf-8")
        if self.vectors is not None:
            np.save(p / "vectors.npy", self.vectors.astype(np.float32), allow_pickle=False)
        elif (p / "vectors.npy").exists():
            (p / "vectors.npy").unlink()
        sources: dict[str, dict] = {}
        for c in self.chunks:
            s = sources.setdefault(c.source, {"n_chunks": 0, "source_date": "", "license": c.license,
                                              "attribution": c.attribution})
            s["n_chunks"] += 1
            s["source_date"] = max(s["source_date"], c.source_date)
        man = {"format": 1, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "n_chunks": len(self.chunks), "embedder": self.embedder.name if self.embedder else None,
               "dim": int(self.vectors.shape[1]) if self.vectors is not None else None,
               "chunks_sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(), "sources": sources}
        (p / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
        self.manifest = man
        return man

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder | None = None) -> "Index":
        p = Path(path)
        man = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
        raw = (p / "chunks.jsonl").read_bytes()
        if hashlib.sha256(raw).hexdigest() != man["chunks_sha256"]:
            raise ValueError("chunks.jsonl does not match its manifest (corrupted or modified). Rebuild the index.")
        chunks = [Chunk.from_dict(json.loads(l)) for l in raw.decode("utf-8").splitlines() if l.strip()]
        if len(chunks) != man["n_chunks"]:
            raise ValueError("chunk count mismatch")
        vectors = None
        if man.get("embedder"):
            if embedder is None:
                pass  # dense vectors exist but no embedder supplied: run BM25-only
            elif embedder.name != man["embedder"]:
                raise ValueError(f"index was built with embedder {man['embedder']!r}, got {embedder.name!r}")
            else:
                vectors = np.load(p / "vectors.npy", allow_pickle=False)
                if vectors.shape[0] != len(chunks):
                    raise ValueError("vector count mismatch")
        return cls(chunks, vectors, embedder, man)

    # ---- search ----------------------------------------------------------------------------------------
    def search(self, query: str, k: int = 10, topics: set[str] | None = None) -> list[Hit]:
        bm = self.bm25.scores(query)
        bm_rank = {i: r for r, i in enumerate(sorted(bm, key=bm.get, reverse=True)[:POOL])}
        dense = None
        dn_rank: dict[int, int] = {}
        if self.vectors is not None and self.embedder is not None:
            q = self.embedder.embed([query], "query")[0]
            dense = self.vectors @ q
            dn_rank = {int(i): r for r, i in enumerate(np.argsort(-dense)[:POOL])}
        tagged: set[int] = set()
        for t in topics or ():
            tagged |= self.by_topic.get(t, set())
        pool = set(bm_rank) | set(dn_rank) | tagged
        hits = []
        for i in pool:
            rrf = (1 / (RRF_K + bm_rank[i] + 1) if i in bm_rank else 0.0) + \
                  (1 / (RRF_K + dn_rank[i] + 1) if i in dn_rank else 0.0)
            hits.append(Hit(i, self.chunks[i], rrf, bm.get(i, 0.0), float(dense[i]) if dense is not None else None,
                            i in tagged))
        hits.sort(key=lambda h: (not h.tag_match, -h.rrf, h.idx))
        return hits[:k]

    # ---- housekeeping ------------------------------------------------------------------------------------
    def age_days(self, today: date | None = None) -> int | None:
        dates = [s["source_date"] for s in self.manifest.get("sources", {}).values() if s.get("source_date")]
        if not dates:
            return None
        return ((today or date.today()) - date.fromisoformat(min(dates))).days

    def stale(self, max_days: int = 180, today: date | None = None) -> bool:
        a = self.age_days(today)
        return a is None or a > max_days
