"""
Gemini Notebook → Zotero.

Three flows, by what they need:

* ``import_new_sources`` — sources the researcher added *inside* Gemini Notebook
  (web pages, YouTube, Discover results) become Zotero items. Works on the
  official API alone (title + original URL); the experimental reader, if
  enabled, adds the extracted full text as a child note.
* ``pull_notes`` — EXPERIMENTAL (needs a NotebookReader). Notes attach to a
  Zotero item only when they unambiguously cite exactly one item by its
  "Author (Year)" label; anything else is collected into one note on the
  collection. Never guessed, never silently dropped (v1 did both).
* ``download_originals`` — fetch original files for web sources via the
  official ``originalUrl``.
"""

from __future__ import annotations

import html
import logging
import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import httpx

from ..notebook.base import NotebookReader, NotebookWriter, Source, SourceState
from ..state import SyncState
from ..zotero import render
from ..zotero.client import ZoteroItem, ZoteroLibrary

logger = logging.getLogger(__name__)


@dataclass
class ReverseReport:
    imported: int = 0
    attached: int = 0
    to_collection_note: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        bits = []
        if self.imported:
            bits.append(f"{self.imported} source(s) imported as Zotero items")
        if self.attached:
            bits.append(f"{self.attached} note(s) attached to their Zotero items")
        if self.to_collection_note:
            bits.append(f"{self.to_collection_note} note(s) saved to the collection "
                        "(they cite several or no items)")
        if self.skipped:
            bits.append(f"{self.skipped} already done")
        if self.errors:
            bits.append(f"{len(self.errors)} error(s)")
        return ", ".join(bits) or "Nothing new"


def _source_kind(src: Source) -> str:
    url = (src.original_url or "").lower()
    if "youtube.com/" in url or "youtu.be/" in url:
        return "youtube"
    if url:
        return "web_page"
    mime = (src.mime_type or "").lower()
    if mime == "application/pdf":
        return "pdf"
    if mime.startswith("image/"):
        return "image"
    return "pasted_text" if mime.startswith("text/") else "docx"


def import_new_sources(notebook_name: str, notebook_title: str, writer: NotebookWriter,
                       zot: ZoteroLibrary, state: SyncState,
                       reader: Optional[NotebookReader] = None,
                       progress: Callable[[str], None] = lambda m: None) -> ReverseReport:
    rep = ReverseReport()
    ours = {s.source_name for s in state.sources_in_notebook(notebook_name)}
    ours |= {o.source_name for o in state.orphans(include_resolved=True)}
    our_names = state.display_names_in_notebook(notebook_name)
    coll_key = zot.find_or_create_collection(f"From Gemini Notebook: {notebook_title}")
    if not coll_key:
        rep.errors.append("Couldn't create the Zotero collection.")
        return rep
    for src in writer.list_sources(notebook_name):
        if src.name in ours or src.display_name in our_names or src.state != SourceState.ACTIVE:
            continue   # CiteBridge put it there (or a copy of it), or it isn't usable
        if state.source_imported(src.name):
            rep.skipped += 1
            continue
        title = src.display_name or src.original_url or "Untitled source"
        progress(f"Importing {title}")
        fulltext = ""
        if reader is not None:
            try:
                fulltext = reader.get_source_text(notebook_name.rsplit("/", 1)[-1], src.id).content
            except Exception as e:
                logger.warning(f"Full text unavailable for {title}: {e}")
        try:
            key = zot.create_item_from_source(
                title=title, url=src.original_url, kind=_source_kind(src), fulltext=fulltext,
                collection_key=coll_key, tags=["citebridge", "from-gemini-notebook"])
        except Exception as e:
            rep.errors.append(f"{title}: {e}")
            continue
        if key:
            state.record_source_import(src.name, key)
            rep.imported += 1
    state.log("import_sources", "success" if not rep.errors else "partial", rep.summary())
    return rep


def _label_index(items: List[ZoteroItem]) -> Dict[str, List[str]]:
    """'smith (2023)' → [item keys] — ambiguous labels map to several keys."""
    idx: Dict[str, List[str]] = {}
    for i in items:
        if render.authors_short(i):   # "(2023)" alone would match far too much
            idx.setdefault(render.author_year(i).lower(), []).append(i.key)
    return idx


def attribute_note(text: str, idx: Dict[str, List[str]]) -> Optional[str]:
    """Return the single Zotero key a note unambiguously refers to, else None."""
    low = text.lower()
    hits = set()
    for label, keys in idx.items():
        # Whole-label match: "Goldsmith (2023)" must not match "Smith (2023)", and the
        # second author in "Smith & Jones (2023)" / "Smith and Jones" isn't "Jones (2023)".
        if re.search(r"(?<![\w&])(?<!& )(?<!and )" + re.escape(label), low):
            hits.update(keys)
    return hits.pop() if len(hits) == 1 else None


def pull_notes(collection_key: str, collection_name: str, reader: NotebookReader,
               zot: ZoteroLibrary, state: SyncState,
               progress: Callable[[str], None] = lambda m: None) -> ReverseReport:
    rep = ReverseReport()
    idx = _label_index(zot.collection_items(collection_key))
    leftovers: List[str] = []
    for nb in state.notebooks_for(collection_key):
        try:
            notes = reader.list_notes(nb.notebook_id)
        except Exception as e:
            rep.errors.append(f"{nb.display_name}: {e}")
            continue
        for note in notes:
            if state.note_imported(note.id):
                rep.skipped += 1
                continue
            body = html.escape(note.content).replace("\n", "<br>")
            key = attribute_note(f"{note.title}\n{note.content}", idx)
            progress(f"Note: {note.title}")
            if not key:
                leftovers.append((nb.notebook_id, note.id,
                                  f"<h3>{html.escape(note.title)}</h3><p>{body}</p>"))
                continue
            try:
                nk = zot.create_note(key, f"Gemini Notebook: {note.title}", body,
                                     tags=["citebridge", "gemini-notebook", "ai-generated"])
            except Exception as e:
                rep.errors.append(f"{note.title}: {e}")
                continue
            if not nk:   # pyzotero reports some failures without raising
                rep.errors.append(f"{note.title}: Zotero didn't save the note; it will be retried.")
                continue
            state.record_note_import(nb.notebook_id, note.id, key, nk)
            rep.attached += 1
    if leftovers:
        try:
            nk = zot.create_note(None, f"Gemini Notebook notes — {collection_name}",
                                 "<p>Notes that cite several sources (or none). Imported by "
                                 "CiteBridge.</p>" + "".join(h for _, _, h in leftovers),
                                 tags=["citebridge", "gemini-notebook", "ai-generated"],
                                 collection_key=collection_key)
        except Exception as e:
            nk = None
            rep.errors.append(f"Collection note: {e}")
        if nk:
            for nb_id, note_id, _ in leftovers:
                state.record_note_import(nb_id, note_id, "", nk)
            rep.to_collection_note += len(leftovers)
        elif not any(e.startswith("Collection note") for e in rep.errors):
            rep.errors.append("Collection note: Zotero didn't save it; it will be retried.")
    state.log("pull_notes", "success" if not rep.errors else "partial", rep.summary())
    return rep


def _safe_name(name: str, ext: str) -> str:
    base = re.sub(r'[<>:"/\\|?*]', "_", name).strip(". ")[:150] or "untitled"
    return base if base.lower().endswith(ext) else base + ext


def download_originals(notebook_name: str, notebook_title: str, writer: NotebookWriter,
                       out_dir: str, timeout: float = 30.0,
                       progress: Callable[[str], None] = lambda m: None) -> ReverseReport:
    """Save the original files behind a notebook's web sources."""
    rep = ReverseReport()
    target = Path(out_dir) / _safe_name(notebook_title, "")
    target.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=timeout, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0 (compatible; CiteBridge/2.0)"}) as http:
        for src in writer.list_sources(notebook_name):
            if not src.original_url or "youtube.com" in src.original_url:
                rep.skipped += 1
                continue
            try:
                r = http.get(src.original_url)
                r.raise_for_status()
                ctype = r.headers.get("content-type", "").split(";")[0]
                ext = mimetypes.guess_extension(ctype) or ".html"
                path = target / _safe_name(src.display_name or src.id, ext)
                n = 2
                while path.exists():
                    path = target / _safe_name(f"{src.display_name or src.id} ({n})", ext)
                    n += 1
                path.write_bytes(r.content)
                rep.imported += 1
                progress(f"Saved {path.name}")
            except Exception as e:
                rep.errors.append(f"{src.display_name}: {e}")
    return rep
