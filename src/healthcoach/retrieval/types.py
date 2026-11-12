from __future__ import annotations
from dataclasses import dataclass, field, asdict


@dataclass
class Chunk:
    id: str
    text: str                                   # what the LLM and the user see
    title: str
    source: str                                 # e.g. "MedlinePlus"
    url: str = ""
    kind: str = "overview"                      # overview | seek_care
    topics: list[str] = field(default_factory=list)    # symptom groups this chunk is tagged for
    aliases: list[str] = field(default_factory=list)   # "also called" / see-references (searchable only)
    license: str = ""
    attribution: str = ""                       # text the UI must show when this chunk is used
    source_date: str = ""                       # ISO date of the source file / review

    @property
    def index_text(self) -> str:
        """Text used for search: title and aliases give short chunks their context."""
        return f"{self.title}. {' '.join(self.aliases)}. {self.text}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


@dataclass
class Passage:
    sid: str                 # "S1", "S2", ... the id the LLM must cite
    chunk: Chunk
    matched_symptom: str
    score: float

    def citation(self) -> dict:
        c = self.chunk
        return {"id": self.sid, "title": c.title, "source": c.source, "url": c.url,
                "attribution": c.attribution, "license": c.license}
