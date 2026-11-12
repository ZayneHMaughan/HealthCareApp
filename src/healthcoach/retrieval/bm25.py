from __future__ import annotations
import math
from collections import Counter, defaultdict
from .textproc import tokenize


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.len = []
        for i, d in enumerate(docs):
            toks = tokenize(d)
            self.len.append(len(toks))
            for t, c in Counter(toks).items():
                self.post[t].append((i, c))
        self.avg = (sum(self.len) / self.n) if self.n else 0.0

    def scores(self, query: str) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for t in set(tokenize(query)):
            plist = self.post.get(t)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, tf in plist:
                denom = tf + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1))
                out[i] += idf * tf * (self.k1 + 1) / denom
        return dict(out)
