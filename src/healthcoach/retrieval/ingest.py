"""Corpus ingestion.

MedlinePlus: only <full-summary> text of English health topics is ingested. MedlinePlus states that health
topic summaries are public domain, but that A.D.A.M. encyclopedia articles, ASHP drug monographs and most
images are copyrighted and may not be ingested into health IT systems without a license. The <site>
records (third-party links) are never read. Topics whose summary carries a copyright marker are skipped
and reported. Re-check https://medlineplus.gov/about/using/usingcontent/ before shipping.

Only parse files you downloaded yourself from medlineplus.gov (see https://medlineplus.gov/xml.html).
"""
from __future__ import annotations
import csv
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from .textproc import chunk_blocks, html_to_blocks
from .types import Chunk

MPLUS_SOURCE = "MedlinePlus"
MPLUS_ATTRIBUTION = "Source: MedlinePlus, National Library of Medicine"
MPLUS_LICENSE = "US government work (public domain); health topic summaries only"

SEEK_RE = re.compile(
    r"\b(call 911|emergency (room|care|department|services|medical)|seek (emergency|medical|immediate)|"
    r"get (medical|emergency) (help|care)|go to the (er|emergency)|"
    r"(see|call|contact|talk to) (a|your) (doctor|health care provider|provider|health care professional))", re.I)
COPYRIGHT_RE = re.compile(r"A\.D\.A\.M|©|\(c\)\s*\d{4}|copyright", re.I)


def classify_kind(text: str) -> str:
    return "seek_care" if SEEK_RE.search(text) else "overview"


def load_topic_map(path: str | Path) -> dict[str, list[str]]:
    """CSV columns: group,title,status. Returns {lowercased MedlinePlus title: [symptom groups]}."""
    out: dict[str, list[str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["title"].strip().lower(), []).append(row["group"].strip())
    return out


def _iso(s: str) -> str:
    for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def _inner(el) -> str:
    """Summary HTML whether the file stores it escaped (text) or as child elements."""
    return (el.text or "") + "".join(ET.tostring(c, encoding="unicode") for c in el)


def parse_medlineplus(path: str | Path, topic_map: dict[str, list[str]] | None = None,
                      max_words: int = 160) -> tuple[list[Chunk], dict]:
    path = Path(path)
    with open(path, "rb") as f:
        if b"<!ENTITY" in f.read(8192):
            raise ValueError("Unexpected entity declarations in XML; refusing to parse.")
    topic_map = topic_map or {}
    chunks: list[Chunk] = []
    rep = {"n_topics": 0, "n_chunks": 0, "skipped_non_english": 0, "skipped_no_summary": 0,
           "skipped_copyright_marker": [], "source_date": ""}
    seen_titles: set[str] = set()
    for event, el in ET.iterparse(path, events=("start", "end")):
        if event == "start" and el.tag == "health-topics":
            rep["source_date"] = _iso(el.attrib.get("date-generated") or el.attrib.get("dategenerated") or "")
        elif event == "end" and el.tag == "health-topic":
            title = (el.attrib.get("title") or (el.text or "")).strip()
            if el.attrib.get("language", "English") != "English":
                rep["skipped_non_english"] += 1
            else:
                summ = el.find("full-summary")
                blocks = html_to_blocks(_inner(summ)) if summ is not None else []
                if not blocks:
                    rep["skipped_no_summary"] += 1
                elif COPYRIGHT_RE.search(" ".join(blocks)):
                    rep["skipped_copyright_marker"].append(title)
                else:
                    aliases = [a.text.strip() for a in el.findall("also-called") if a.text] + \
                              [a.text.strip() for a in el.findall("see-reference") if a.text]
                    seen_titles.add(title.lower())
                    tid = el.attrib.get("id", str(rep["n_topics"]))
                    for n, text in enumerate(chunk_blocks(blocks, max_words)):
                        chunks.append(Chunk(
                            id=f"mplus-{tid}-{n}", text=text, title=title, source=MPLUS_SOURCE,
                            url=el.attrib.get("url", ""), kind=classify_kind(text),
                            topics=list(topic_map.get(title.lower(), [])), aliases=aliases,
                            license=MPLUS_LICENSE, attribution=MPLUS_ATTRIBUTION,
                            source_date=rep["source_date"]))
                    rep["n_topics"] += 1
            el.clear()
    rep["n_chunks"] = len(chunks)
    rep["map_titles_unmatched"] = sorted(set(topic_map) - seen_titles)
    rep["tagged_groups"] = sorted({g for c in chunks for g in c.topics})
    return chunks, rep


def _front_matter(text: str, name: str) -> tuple[dict, str]:
    m = re.match(r"---\n(.*?)\n---\n?(.*)", text, re.S)
    if not m:
        raise ValueError(f"{name}: missing front matter")
    fm = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip()
    return fm, m.group(2)


def parse_curated_dir(path: str | Path, max_words: int = 160) -> list[Chunk]:
    """Hand-authored or licensed content as Markdown with front matter:
         ---
         title: ...   source: ...   license: ...   url: ...   topics: [chest_pain, fainting]
         reviewed: 2026-01-31   attribution: ...
         ---
         ## When to get care
         ...
    Each '## ' section is chunked separately; sections whose heading mentions 'when to', 'seek' or
    'emergency' are marked seek_care."""
    out: list[Chunk] = []
    for f in sorted(Path(path).glob("*.md")):
        fm, body = _front_matter(f.read_text(encoding="utf-8"), f.name)
        missing = [k for k in ("title", "source", "license") if not fm.get(k)]
        if missing:
            raise ValueError(f"{f.name}: missing front matter fields {missing}")
        topics = [t.strip() for t in fm.get("topics", "").strip("[]").split(",") if t.strip()]
        reviewed = fm.get("reviewed", "")
        if reviewed:
            datetime.strptime(reviewed, "%Y-%m-%d")
        parts = re.split(r"(?m)^##\s+(.*)$", body)
        sections = [("", parts[0])] + list(zip(parts[1::2], parts[2::2]))
        for si, (heading, text) in enumerate(sections):
            if not text.strip():
                continue
            for ci, chunk in enumerate(chunk_blocks(html_to_blocks(text), max_words)):
                kind = "seek_care" if re.search(r"when to|seek|emergency", heading, re.I) else classify_kind(chunk)
                out.append(Chunk(
                    id=f"cur-{f.stem}-{si}-{ci}", text=(f"{heading}\n{chunk}" if heading else chunk),
                    title=fm["title"], source=fm["source"], url=fm.get("url", ""), kind=kind, topics=topics,
                    license=fm["license"], attribution=fm.get("attribution", fm["source"]), source_date=reviewed))
    return out
