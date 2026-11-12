"""Turns an Intake (+ tier) into a small, cited, budgeted set of passages for the LLM.

Design rules:
 - one query per selected symptom, so a second symptom can't crowd out the first (round-robin selection)
 - a passage must be *relevant* (tagged for the symptom, or title/alias matches the symptom, or dense
   similarity above a threshold). BM25 alone never admits a passage: weak matches are rejected on purpose.
 - nothing relevant -> the symptom is reported as `uncovered`; the pipeline then skips the LLM
 - retrieval can add context but never changes the urgency tier
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from ..schema import Intake, Tier
from .index import Hit, Index
from .textproc import est_tokens, tokenize
from .types import Passage

LABELS = Path(__file__).parent.parent / "intake" / "data" / "symptoms.json"

MODIFIER_PHRASES = {
    "sudden_onset": "sudden onset", "worst_ever": "worst headache", "stiff_neck": "stiff neck",
    "radiating_arm_jaw": "pain spreading to arm or jaw", "sweating": "sweating", "at_rest": "short of breath at rest",
    "throat_tightness": "throat swelling", "one_leg": "swelling in one leg", "rigid_abdomen": "rigid abdomen",
    "heavy_bleeding": "heavy bleeding", "unable_keep_fluids": "cannot keep fluids down",
    "loss_of_consciousness": "loss of consciousness", "leg_weakness": "leg weakness",
    "bladder_bowel_loss": "loss of bladder or bowel control",
}


@dataclass
class RetrievalResult:
    passages: list[Passage] = field(default_factory=list)
    uncovered: list[str] = field(default_factory=list)

    @property
    def ids(self) -> list[str]:
        return [p.sid for p in self.passages]

    def prompt_text(self) -> str:
        return "\n\n".join(f"[{p.sid}] {p.chunk.title} ({p.chunk.source})\n{p.chunk.text}" for p in self.passages)

    def citations(self, cited: list[str] | None = None) -> list[dict]:
        return [p.citation() for p in self.passages if cited is None or p.sid in cited]


_GENERIC = frozenset({"pain", "problem", "disorder", "disease", "injur", "symptom", "infection", "common"})


def _name_matches(symptom_tokens: list[str], name_tokens: list[str], min_ratio: float) -> bool:
    """A topic title/alias matches a symptom if enough of its tokens appear in the symptom's words AND at
    least one shared token is specific ("pain" alone must never link Chest Pain to Testicle Pain)."""
    names = set(name_tokens)
    shared = set(symptom_tokens) & names
    return bool(names) and len(shared) / len(names) >= min_ratio and bool(shared - _GENERIC)


class Retriever:
    def __init__(self, index: Index, per_symptom: int = 2, budget_tokens: int = 1100, min_dense: float = 0.55,
                 title_overlap: float = 0.5, pool: int = 12, labels_path: Path = LABELS):
        self.index, self.per_symptom, self.budget = index, per_symptom, budget_tokens
        self.min_dense, self.title_overlap, self.pool = min_dense, title_overlap, pool
        self.labels = json.loads(Path(labels_path).read_text())["groups"]

    def _words(self, g: str) -> list[str]:
        s = self.labels.get(g)
        return [s["label"], *s["synonyms"][:2]] if s else [g.replace("_", " ")]

    def build_query(self, g: str, intake: Intake) -> str:
        mods = [MODIFIER_PHRASES[m] for m in sorted(intake.modifiers) if m in MODIFIER_PHRASES][:3]
        extra = ["pregnancy"] if intake.pregnant else []
        return " ".join(self._words(g) + mods + extra)

    def _relevant(self, h: Hit, g: str) -> bool:
        if h.tag_match:
            return True
        if h.dense is not None and h.dense >= self.min_dense:
            return True
        sym = tokenize(" ".join(self._words(g)))
        return any(_name_matches(sym, tokenize(n), self.title_overlap) for n in [h.chunk.title, *h.chunk.aliases])

    def retrieve(self, intake: Intake, tier: Tier = Tier.SEE_DOCTOR_SOON) -> RetrievalResult:
        per: dict[str, list[Hit]] = {}
        for g in sorted(intake.symptoms):
            hits = self.index.search(self.build_query(g, intake), k=self.pool, topics={g})
            ok = [h for h in hits if self._relevant(h, g)]
            if tier >= Tier.URGENT:   # urgent: lead with "when to get care" passages
                ok = [h for h in ok if h.chunk.kind == "seek_care"] + [h for h in ok if h.chunk.kind != "seek_care"]
            per[g] = ok

        queues = {g: list(v) for g, v in per.items()}
        selected: list[Hit] = []
        seen: set[str] = set()
        used = 0
        for _ in range(self.per_symptom):
            for g, q in queues.items():
                while q:
                    h = q.pop(0)
                    cost = est_tokens(h.chunk.text) + 12
                    if h.chunk.id in seen or used + cost > self.budget:
                        continue
                    selected.append(h)
                    seen.add(h.chunk.id)
                    used += cost
                    break
        passages = [Passage(f"S{n}", h.chunk, next(g for g, v in per.items() if h in v), h.rrf)
                    for n, h in enumerate(selected, 1)]
        uncovered = [g for g, v in per.items() if not any(h.chunk.id in seen for h in v)]
        return RetrievalResult(passages, uncovered)

    __call__ = retrieve


def coverage_report(retriever: Retriever, groups: list[str] | None = None) -> dict[str, list[str]]:
    """For each symptom group, which passage titles would be retrieved for a bare single-symptom intake.
    Empty list = a gap in the corpus (the app would fall back to a template for that symptom)."""
    groups = groups or sorted(retriever.labels)
    return {g: [p.chunk.title for p in retriever.retrieve(Intake(age=40, symptoms={g})).passages] for g in groups}
