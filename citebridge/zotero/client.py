"""
Zotero access for CiteBridge v2.

Reads prefer the Zotero 7 desktop *local API* (http://localhost:23119/api,
library "0") — no API key needed, fast, and attachment files resolve straight
to disk via ``links.enclosure.href``. When Zotero desktop isn't running we
fall back to the zotero.org web API (needs key + library id).

Writes (notes back from Gemini Notebook, imported sources) always use the web
API: the local API is read-only.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

from pyzotero import zotero

from ..config import ZoteroConfig

logger = logging.getLogger(__name__)


class ZoteroUnavailable(Exception):
    pass


class ZoteroWriteUnavailable(ZoteroUnavailable):
    pass


@dataclass
class ZoteroCollection:
    key: str
    name: str
    parent_key: Optional[str] = None
    num_items: int = 0


@dataclass
class Annotation:
    text: str = ""
    comment: str = ""
    page: str = ""
    color: str = ""


@dataclass
class ZoteroItem:
    key: str
    title: str
    item_type: str
    version: int = 0
    creators: List[Dict[str, str]] = field(default_factory=list)
    date: str = ""
    doi: str = ""
    url: str = ""
    abstract: str = ""
    publication: str = ""
    citation_key: str = ""
    tags: List[str] = field(default_factory=list)
    collections: List[str] = field(default_factory=list)


@dataclass
class PdfAttachment:
    key: str
    filename: str
    file_url: str = ""           # file:// from the local API, when available
    md5: str = ""                # Zotero's hash of the stored file (change detection)
    version: int = 0             # the attachment's own version (fallback when no md5)


def _citation_key(data: Dict[str, Any]) -> str:
    """Zotero 7 citationKey field, else Better BibTeX's 'Citation Key:' line in Extra."""
    if data.get("citationKey"):
        return data["citationKey"]
    for line in (data.get("extra") or "").splitlines():
        if line.lower().startswith("citation key:"):
            return line.split(":", 1)[1].strip()
    return ""


def parse_item(raw: Dict[str, Any]) -> ZoteroItem:
    d = raw.get("data", raw)
    return ZoteroItem(
        key=d.get("key", ""),
        title=d.get("title") or d.get("caseName") or d.get("nameOfAct") or "Untitled",
        item_type=d.get("itemType", ""),
        version=int(d.get("version") or raw.get("version") or 0),
        creators=d.get("creators", []) or [],
        date=d.get("date", ""),
        doi=d.get("DOI", ""),
        url=d.get("url", ""),
        abstract=d.get("abstractNote", ""),
        publication=(d.get("publicationTitle") or d.get("bookTitle") or d.get("proceedingsTitle")
                     or d.get("websiteTitle") or d.get("publisher") or ""),
        citation_key=_citation_key(d),
        tags=[t.get("tag", "") for t in d.get("tags", []) or []],
        collections=d.get("collections", []) or [],
    )


class ZoteroLibrary:
    def __init__(self, cfg: ZoteroConfig):
        self.cfg = cfg
        self._reader: Optional[zotero.Zotero] = None
        self._writer: Optional[zotero.Zotero] = None
        self.read_mode = ""

    # ── connections ──

    def _web(self) -> zotero.Zotero:
        if not (self.cfg.api_key and self.cfg.library_id):
            raise ZoteroWriteUnavailable(
                "A Zotero API key and library ID are needed for this (Settings → Zotero).")
        return zotero.Zotero(self.cfg.library_id, self.cfg.library_type, self.cfg.api_key)

    @property
    def reader(self) -> zotero.Zotero:
        if self._reader is not None:
            return self._reader
        if self.cfg.use_local_api and self.cfg.library_type == "user":
            try:
                z = zotero.Zotero("0", "user", local=True)
                z.collections(limit=1)
                self._reader, self.read_mode = z, "local"
                return z
            except Exception as e:
                logger.info(f"Zotero desktop local API unavailable ({e}); trying web API")
        try:
            self._reader, self.read_mode = self._web(), "web"
        except ZoteroWriteUnavailable:
            raise ZoteroUnavailable(
                "Can't read your Zotero library. Open Zotero 7 desktop (and enable "
                "Settings → Advanced → 'Allow other applications on this computer to "
                "communicate with Zotero'), or add a Zotero API key.")
        return self._reader

    @property
    def writer(self) -> zotero.Zotero:
        if self._writer is None:
            self._writer = self._web()
        return self._writer

    @property
    def read_mode_label(self) -> str:
        return {"local": "Zotero desktop (local)", "web": "zotero.org (web API)"}.get(
            self.read_mode, "not connected")

    def test_connection(self) -> tuple:
        try:
            self._reader = None
            self.reader.collections(limit=1)
            return True, f"Connected via {self.read_mode_label}"
        except Exception as e:
            return False, str(e)

    # ── reads ──

    def collections(self) -> List[ZoteroCollection]:
        out = []
        for c in self.reader.everything(self.reader.collections()):
            d, meta = c.get("data", c), c.get("meta", {}) or {}
            key = d.get("key", "")
            n = meta.get("numItems")
            if n is None:
                try:
                    n = self.reader.num_collectionitems(key)
                except Exception:
                    n = 0
            out.append(ZoteroCollection(key, d.get("name", "Untitled"),
                                        d.get("parentCollection") or None, int(n or 0)))
        return out

    def collection_items(self, collection_key: str) -> List[ZoteroItem]:
        raw = self.reader.everything(self.reader.collection_items_top(collection_key))
        return [parse_item(r) for r in raw
                if r.get("data", r).get("itemType") not in ("attachment", "note", "annotation")]

    def children(self, item_key: str) -> List[Dict[str, Any]]:
        try:
            return self.reader.children(item_key)
        except Exception as e:
            logger.warning(f"Couldn't read children of {item_key}: {e}")
            return []

    def pdf_attachment(self, children: List[Dict[str, Any]]) -> Optional[PdfAttachment]:
        for ch in children:
            d = ch.get("data", ch)
            if d.get("itemType") == "attachment" and d.get("contentType") == "application/pdf" \
                    and d.get("linkMode") != "linked_url":
                href = (ch.get("links", {}).get("enclosure", {}) or {}).get("href", "")
                return PdfAttachment(d.get("key", ""), d.get("filename") or f"{d.get('key')}.pdf",
                                     href if href.startswith("file://") else "", d.get("md5") or "",
                                     int(d.get("version") or ch.get("version") or 0))
        return None

    def notes_html(self, children: List[Dict[str, Any]]) -> List[str]:
        return [ch.get("data", ch).get("note", "") for ch in children
                if ch.get("data", ch).get("itemType") == "note"]

    def annotations(self, attachment_key: str) -> List[Annotation]:
        out = []
        for ch in self.children(attachment_key):
            d = ch.get("data", ch)
            if d.get("itemType") == "annotation" and (d.get("annotationText") or d.get("annotationComment")):
                out.append(Annotation(d.get("annotationText", ""), d.get("annotationComment", ""),
                                      d.get("annotationPageLabel", ""), d.get("annotationColor", "")))
        return out

    def resolve_pdf(self, att: PdfAttachment, download_dir: Optional[str] = None) -> Optional[str]:
        """Local path to the PDF: local-API file URL → Zotero/storage → web download."""
        if att.file_url:
            p = Path(unquote(urlparse(att.file_url).path))
            if p.exists():
                return str(p)
        if self.cfg.local_storage_path:
            d = Path(self.cfg.local_storage_path) / att.key
            if d.is_dir():
                for f in d.iterdir():
                    if f.suffix.lower() == ".pdf":
                        return str(f)
        try:
            data = self._web().file(att.key)
        except ZoteroWriteUnavailable:
            return None
        except Exception as e:
            logger.error(f"Download of {att.key} failed: {e}")
            return None
        dest = Path(download_dir or tempfile.mkdtemp(prefix="citebridge_")) / att.filename
        dest.write_bytes(data)
        return str(dest)

    # ── writes (web API) ──

    def create_note(self, parent_item_key: Optional[str], title: str, html: str,
                    tags: Optional[List[str]] = None,
                    collection_key: Optional[str] = None) -> Optional[str]:
        tpl = self.writer.item_template("note")
        tpl["note"] = f"<h2>{_escape(title)}</h2>\n{html}"
        tpl["tags"] = [{"tag": t} for t in (tags or ["citebridge"])]
        if collection_key and not parent_item_key:
            tpl["collections"] = [collection_key]
        res = (self.writer.create_items([tpl], parentid=parent_item_key) if parent_item_key
               else self.writer.create_items([tpl]))
        return _created_key(res)

    def find_or_create_collection(self, name: str) -> Optional[str]:
        for c in self.writer.everything(self.writer.collections()):
            if c.get("data", c).get("name") == name:
                return c.get("data", c).get("key")
        return _created_key(self.writer.create_collections([{"name": name}]))

    def create_item_from_source(self, *, title: str, url: Optional[str], kind: str,
                                fulltext: str, collection_key: str, tags: List[str]) -> Optional[str]:
        zt = NLM_KIND_TO_ZOTERO.get(kind, "document")
        tpl = self.writer.item_template(zt)
        tpl["title"] = title
        tpl["tags"] = [{"tag": t} for t in tags]
        tpl["collections"] = [collection_key]
        if url and "url" in tpl:
            tpl["url"] = url
        key = _created_key(self.writer.create_items([tpl]))
        if key and fulltext:
            text = fulltext[:500_000] + ("\n\n[…truncated]" if len(fulltext) > 500_000 else "")
            try:   # the item exists either way; a failed note must not cause a re-import
                self.create_note(key, "Full text (from Gemini Notebook)",
                                 f"<pre>{_escape(text)}</pre>", tags=["citebridge-fulltext"])
            except Exception as e:
                logger.warning(f"Item {key} imported, but its full-text note failed: {e}")
        return key


def detect_storage_path() -> str:
    """Best guess at the Zotero/storage folder (ported from v1 utils)."""
    home = Path.home()
    candidates = [home / "Zotero" / "storage"]
    if sys.platform.startswith("linux"):
        candidates += [home / ".zotero" / "zotero" / "storage",
                       home / "snap" / "zotero-snap" / "common" / "Zotero" / "storage"]
    return next((str(p) for p in candidates if p.is_dir()), "")


NLM_KIND_TO_ZOTERO = {
    "web_page": "webpage", "pdf": "document", "youtube": "videoRecording",
    "google_docs": "document", "google_slides": "presentation", "google_spreadsheet": "document",
    "google_drive_audio": "audioRecording", "google_drive_video": "videoRecording",
    "csv": "document", "docx": "document", "markdown": "document", "pasted_text": "document",
    "image": "artwork",
}


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _created_key(result: Any) -> Optional[str]:
    created = (result or {}).get("successful") or {}
    if not created:
        return None
    first = list(created.values())[0]
    return first.get("data", first).get("key") or first.get("key")
