"""
Render Zotero items into what Gemini Notebook sees.

Accuracy is the point of this module. Gemini Notebook cites sources by their
display name, and the API won't give citations back, so:

* every source is named ``Author (Year) — Title`` so citations in the
  notebook are immediately recognisable as specific Zotero items;
* items without a PDF still enter the notebook, as a Markdown "citation card"
  carrying the full reference, abstract and the researcher's own notes;
* each notebook gets one "Bibliography" source listing every item with its
  author-year label, DOI and Zotero key, so answers can be traced back to the
  exact library item (and the model has correct metadata to cite from).
"""

from __future__ import annotations

import hashlib
import re
from html.parser import HTMLParser
from typing import Iterable, List, Optional, Sequence, Tuple

from .client import Annotation, ZoteroItem

MAX_DISPLAY = 500
MAX_TEXT_BYTES = 1_000_000
BIBLIOGRAPHY_PREFIX = "📚 Bibliography — "
ANNOTATIONS_PREFIX = "🖍 Highlights & notes — "


# ── names ──

def _last_name(c: dict) -> str:
    return (c.get("lastName") or c.get("name") or "").strip()


def authors_short(item: ZoteroItem) -> str:
    authors = [c for c in item.creators if c.get("creatorType", "author") == "author"] or item.creators
    names = [n for n in (_last_name(c) for c in authors) if n]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} & {names[1]}"
    return f"{names[0]} et al."


def year(item: ZoteroItem) -> str:
    m = re.search(r"\b(1[5-9]\d\d|2\d\d\d)\b", item.date or "")
    return m.group(1) if m else "n.d."


def author_year(item: ZoteroItem) -> str:
    a = authors_short(item)
    return f"{a} ({year(item)})" if a else f"({year(item)})"


def display_name(item: ZoteroItem) -> str:
    name = f"{author_year(item)} — {item.title.strip()}"
    return name if len(name) <= MAX_DISPLAY else name[:MAX_DISPLAY - 1] + "…"


def part_suffix(part: int, parts: int) -> str:
    return f" ({part}/{parts})" if parts > 1 else ""


def notebook_display_name(collection_name: str, part: int = 1, parts: int = 1) -> str:
    return (collection_name + part_suffix(part, parts))[:MAX_DISPLAY]


# ── text helpers ──

class _Text(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr", "blockquote"}

    def __init__(self):
        super().__init__()
        self.parts: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "li":
            self.parts.append("\n- ")
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _Text()
    p.feed(html or "")
    return re.sub(r"\n{3,}", "\n\n", "".join(p.parts)).strip()


def full_citation(item: ZoteroItem) -> str:
    """A compact APA-like reference line."""
    names = []
    for c in item.creators:
        last, first = _last_name(c), (c.get("firstName") or "").strip()
        if last:
            names.append(f"{last}, {first[0]}." if first else last)
    if len(names) > 1:
        authors = ", ".join(names[:-1]) + ", & " + names[-1]
    else:
        authors = names[0] if names else ""
    bits = [f"{authors} ({year(item)})." if authors else f"({year(item)}).", f"{item.title.strip()}."]
    if item.publication:
        bits.append(f"*{item.publication}*.")
    if item.doi:
        bits.append(f"https://doi.org/{item.doi}")
    elif item.url:
        bits.append(item.url)
    return " ".join(bits)


def fit(text: str, limit: int = MAX_TEXT_BYTES) -> str:
    """Keep Markdown within the API's 1 MB text-source limit."""
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text
    note = "\n\n_[CiteBridge: truncated to fit Gemini Notebook's 1 MB text-source limit.]_\n"
    return data[: limit - len(note.encode())].decode("utf-8", "ignore") + note


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── sources ──

def citation_card(item: ZoteroItem, notes_html: Sequence[str] = (),
                  annotations: Sequence[Annotation] = ()) -> str:
    """Markdown source for an item that has no PDF to upload."""
    rows = [("Reference", full_citation(item)), ("Cite as", author_year(item)),
            ("Item type", item.item_type), ("Zotero key", f"`{item.key}`")]
    if item.citation_key:
        rows.append(("Citation key", f"`{item.citation_key}`"))
    if item.doi:
        rows.append(("DOI", item.doi))
    if item.url:
        rows.append(("URL", item.url))
    if item.tags:
        rows.append(("Tags", ", ".join(item.tags)))

    out = [f"# {item.title.strip()}", "",
           "> Bibliographic record exported from the researcher's Zotero library by CiteBridge. "
           "No full text was available; cite this item as "
           f"**{author_year(item)}**.", ""]
    out += [f"- **{k}:** {v}" for k, v in rows]
    if item.abstract:
        out += ["", "## Abstract", "", item.abstract.strip()]
    texts = [t for t in (html_to_text(h) for h in notes_html) if t]
    if texts:
        out += ["", "## Researcher's notes"]
        for t in texts:
            out += ["", t]
    if annotations:
        out += ["", "## Highlights"] + [_annotation_line(a) for a in annotations]
    return fit("\n".join(out) + "\n")


def _annotation_line(a: Annotation) -> str:
    page = f" (p. {a.page})" if a.page else ""
    line = f"- “{a.text.strip()}”{page}" if a.text else f"- {page.strip()}"
    if a.comment:
        line += f"\n  - Note: {a.comment.strip()}"
    return line


def bibliography(collection_name: str, items: Iterable[ZoteroItem],
                 part: int = 1, parts: int = 1) -> Tuple[str, str]:
    """(display_name, markdown) for the notebook's Bibliography source."""
    items = sorted(items, key=lambda i: (authors_short(i).lower(), year(i), i.title.lower()))
    title = f"{BIBLIOGRAPHY_PREFIX}{collection_name}{part_suffix(part, parts)}"
    out = [f"# Bibliography: {collection_name}{part_suffix(part, parts)}", "",
           "This is the authoritative reference list for the sources in this notebook, exported "
           "from the researcher's Zotero library by CiteBridge. Each source in this notebook is "
           "named **Author (Year) — Title**. When answering, cite sources with that author-year "
           "label and use the details below (authors, year, venue, DOI) rather than inferring them.",
           "", f"{len(items)} items.", ""]
    for i in items:
        out.append(f"- **{author_year(i)}** — {full_citation(i)}  \n  Zotero key: `{i.key}`"
                   + (f" · Citation key: `{i.citation_key}`" if i.citation_key else ""))
    return title[:MAX_DISPLAY], fit("\n".join(out) + "\n")


def annotations_digest(collection_name: str,
                       per_item: Iterable[Tuple[ZoteroItem, Sequence[str], Sequence[Annotation]]],
                       part: int = 1, parts: int = 1) -> Optional[Tuple[str, str]]:
    """(display_name, markdown) gathering the researcher's notes + highlights, or None if empty."""
    sections = []
    for item, notes_html, annots in per_item:
        texts = [t for t in (html_to_text(h) for h in notes_html) if t]
        if not texts and not annots:
            continue
        sec = [f"## {author_year(item)} — {item.title.strip()}", f"Zotero key: `{item.key}`"]
        for t in texts:
            sec += ["", t]
        if annots:
            sec += [""] + [_annotation_line(a) for a in annots]
        sections.append("\n".join(sec))
    if not sections:
        return None
    title = f"{ANNOTATIONS_PREFIX}{collection_name}{part_suffix(part, parts)}"
    head = [f"# Researcher's highlights and notes: {collection_name}", "",
            "Exported from Zotero by CiteBridge. These are the researcher's own annotations "
            "and notes on the sources in this notebook, grouped by source.", ""]
    return title[:MAX_DISPLAY], fit("\n".join(head) + "\n\n".join(sections) + "\n")
