"""Tokenizing, HTML-to-text, and chunking. Standard library only."""
from __future__ import annotations
import re
from html.parser import HTMLParser

_STOP = frozenset("a an and are as at be by for from has have in is it its of on or that the this to was were will with "
                  "you your can may also often more most other some such than then there these they what when which who".split())


def stem(w: str) -> str:
    """Very light suffix stripper: enough to match vomit/vomiting, ache/aches."""
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def tokenize(s: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _STOP]


def est_tokens(text: str) -> int:
    """Rough LLM token estimate (words * 1.35). Good enough for budgeting a small context window."""
    return int(len(text.split()) * 1.35) + 4


class _Blocks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._buf: list[str] = []
        self._items: list[str] = []
        self._in_li = False

    def _flush_text(self):
        t = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        self._buf = []
        return t

    def handle_starttag(self, tag, attrs):
        if tag in ("p", "div", "h1", "h2", "h3", "h4", "ul", "ol"):
            t = self._flush_text()
            if t:
                self.blocks.append(t)
        elif tag == "li":
            self._flush_text()
            self._in_li = True
        elif tag == "br":
            self._buf.append(" ")

    def handle_endtag(self, tag):
        if tag == "li":
            t = self._flush_text()
            if t:
                self._items.append("- " + t)
            self._in_li = False
        elif tag in ("ul", "ol"):
            if self._items:
                self.blocks.append("\n".join(self._items))
                self._items = []
        elif tag in ("p", "div", "h1", "h2", "h3", "h4"):
            t = self._flush_text()
            if t:
                self.blocks.append(t)

    def handle_data(self, data):
        self._buf.append(data)

    def close(self):
        super().close()
        t = self._flush_text()
        if t:
            self.blocks.append(t)
        if self._items:
            self.blocks.append("\n".join(self._items))


def html_to_blocks(html: str) -> list[str]:
    if "<" not in html:
        return [re.sub(r"[ \t]+", " ", b).strip() for b in re.split(r"\n\s*\n", html) if b.strip()]
    p = _Blocks()
    p.feed(html)
    p.close()
    return p.blocks


def chunk_blocks(blocks: list[str], max_words: int = 160) -> list[str]:
    """Pack blocks into chunks of at most ~max_words, never separating a lead-in
    sentence ("Symptoms may include") from the list that follows it."""
    units: list[str] = []
    i = 0
    while i < len(blocks):
        cur = blocks[i]
        while (i + 1 < len(blocks) and not cur.startswith("- ") and not re.search(r"[.!?\"”)]$", cur)):
            i += 1
            cur = cur + "\n" + blocks[i]
        units.append(cur)
        i += 1

    pieces: list[str] = []
    for u in units:
        if len(u.split()) <= max_words:
            pieces.append(u)
            continue
        parts = u.split("\n") if "\n- " in u or u.startswith("- ") else re.split(r"(?<=[.!?])\s+", u)
        buf: list[str] = []
        for part in parts:
            if buf and len(" ".join(buf + [part]).split()) > max_words:
                pieces.append("\n".join(buf) if "\n" in u else " ".join(buf))
                buf = []
            buf.append(part)
        if buf:
            pieces.append("\n".join(buf) if "\n" in u else " ".join(buf))

    chunks: list[str] = []
    cur = ""
    for p in pieces:
        if cur and len((cur + "\n\n" + p).split()) > max_words:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks
