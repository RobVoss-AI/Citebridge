"""
Sync planner: decide what to send to Gemini Notebook for one Zotero collection.

Pure logic — no network or Zotero I/O — so it is fully unit-testable. The UI
shows the plan before anything runs.

Constraints the planner respects:
* Per-notebook source caps by plan tier. A collection that doesn't fit is
  split into "Name (1/2)", "Name (2/2)"… notebooks. Items are *sticky*: once
  in a notebook they stay there.
* Nothing can be deleted through the API. A replaced/removed source keeps
  occupying a slot until the user deletes it by hand, so capacity is counted
  from the live notebook when we have it.
* Every item with any usable content gets in: PDF → upload; web-native items
  with a public URL → web source; everything else → Markdown citation card.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..notebook.base import Source, SourceState
from ..state import (ANNOTATIONS, BIBLIOGRAPHY, CARD, COLLECTION_LEVEL_KINDS, PDF, URL,
                     NotebookRecord, SourceRecord)
from ..zotero import render
from ..zotero.client import Annotation, PdfAttachment, ZoteroItem

# Item types whose "content" really is the web page at their URL.
WEB_NATIVE_TYPES = {"webpage", "blogPost", "forumPost", "videoRecording", "podcast",
                    "encyclopediaArticle", "newspaperArticle", "magazineArticle"}
URL_FAILURES_THAT_FALL_BACK = {3, 7, 9, 13}
RETRYABLE_FAILURES = {8, 13}          # source limit (room may have been freed), internal error


@dataclass
class ItemMaterial:
    item: ZoteroItem
    pdf: Optional[PdfAttachment] = None
    notes_html: List[str] = field(default_factory=list)
    annotations: List[Annotation] = field(default_factory=list)


@dataclass
class Action:
    source_kind: str                  # pdf | url | card | bibliography | annotations
    part: int
    display_name: str
    reason: str                       # new | changed | missing | fallback
    content_hash: str = ""
    zotero_key: Optional[str] = None
    zotero_version: Optional[int] = None
    pdf: Optional[PdfAttachment] = None
    url: Optional[str] = None
    text: Optional[str] = None
    replaces: Optional[str] = None    # source_name to supersede


@dataclass
class NotebookTarget:
    part: int
    display_name: str
    existing: Optional[NotebookRecord] = None
    rename: bool = False              # existing notebook needs its "(k/n)" name updated


@dataclass
class Skip:
    title: str
    reason: str
    zotero_key: Optional[str] = None


@dataclass
class OrphanAction:
    source_name: str
    reason: str


@dataclass
class CollectionPlan:
    collection_key: str
    collection_name: str
    notebooks: List[NotebookTarget] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    skips: List[Skip] = field(default_factory=list)
    orphans: List[OrphanAction] = field(default_factory=list)
    resolved_orphans: List[str] = field(default_factory=list)   # user deleted them in the UI
    unchanged: int = 0
    warnings: List[str] = field(default_factory=list)
    materials: Dict[str, ItemMaterial] = field(default_factory=dict, repr=False)

    @property
    def parts(self) -> int:
        return len(self.notebooks)

    @property
    def new_notebooks(self) -> int:
        return sum(1 for n in self.notebooks if n.existing is None)

    @property
    def is_noop(self) -> bool:
        return not (self.actions or self.orphans or self.new_notebooks
                    or any(n.rename for n in self.notebooks))

    def count(self, *kinds: str) -> int:
        return sum(1 for a in self.actions if a.source_kind in kinds)

    def summary(self) -> str:
        bits = []
        if self.new_notebooks:
            bits.append(f"{self.new_notebooks} new notebook(s)")
        if self.count(PDF):
            bits.append(f"{self.count(PDF)} PDF(s)")
        if self.count(URL):
            bits.append(f"{self.count(URL)} web page(s)")
        if self.count(CARD):
            bits.append(f"{self.count(CARD)} citation card(s)")
        if self.count(BIBLIOGRAPHY, ANNOTATIONS):
            bits.append(f"{self.count(BIBLIOGRAPHY, ANNOTATIONS)} bibliography/highlights source(s)")
        if self.unchanged:
            bits.append(f"{self.unchanged} already up to date")
        if self.skips:
            bits.append(f"{len(self.skips)} skipped")
        if self.orphans:
            bits.append(f"{len(self.orphans)} to clean up")
        return ", ".join(bits) or "Nothing to do"


# ── item → desired source ──

def desired_source(m: ItemMaterial, cite_without_pdf: bool) -> Optional[Action]:
    item = m.item
    name = render.display_name(item)
    if m.pdf is not None:
        # md5 for stored files; otherwise the attachment's own version, so editing the
        # parent item's tags/metadata doesn't re-upload the PDF.
        h = f"md5:{m.pdf.md5}" if m.pdf.md5 else f"av:{m.pdf.version}:{m.pdf.key}"
        return Action(PDF, 0, name, "new", h, item.key, item.version, pdf=m.pdf)
    if not cite_without_pdf:
        return None
    url = (item.url or "").strip()
    if url.lower().startswith("https://") and (item.item_type in WEB_NATIVE_TYPES
                                               or "youtube.com/" in url or "youtu.be/" in url):
        return Action(URL, 0, name, "new", f"url:{url}", item.key, item.version, url=url)
    text = render.citation_card(item, m.notes_html, m.annotations)
    return Action(CARD, 0, name, "new", render.content_hash(text), item.key, item.version, text=text)


def card_for(m: ItemMaterial, part: int, replaces: Optional[str],
             basis_hash: Optional[str] = None) -> Action:
    """Citation card. ``basis_hash`` records the URL it stands in for, so the next sync
    doesn't try (and fail) that URL again."""
    text = render.citation_card(m.item, m.notes_html, m.annotations)
    h = render.content_hash(text)
    if basis_hash:
        h = f"fallback|{basis_hash}|{h}"
    return Action(CARD, part, render.display_name(m.item), "fallback", h,
                  m.item.key, m.item.version, text=text, replaces=replaces)


# ── planning ──

def plan_collection(
    collection_key: str,
    collection_name: str,
    materials: Sequence[ItemMaterial],
    *,
    existing_notebooks: Sequence[NotebookRecord],
    tracked_sources: Sequence[SourceRecord],
    unresolved_orphans: Dict[str, List[str]],          # notebook_name → orphan source names
    live_sources: Optional[Dict[str, List[Source]]],   # notebook_name → live sources (None = unknown)
    cap: int,
    include_bibliography: bool = True,
    include_annotations: bool = True,
    cite_without_pdf: bool = True,
) -> CollectionPlan:
    plan = CollectionPlan(collection_key, collection_name)
    plan.materials = {m.item.key: m for m in materials}
    if not materials and not existing_notebooks:
        plan.warnings.append("This collection has no items yet.")
        return plan
    # One slot per enabled collection-level source is held back for it (or for its
    # replacement — the old version stays until the user deletes it).
    reserved = int(include_bibliography) + int(include_annotations)
    item_limit = cap - reserved
    if item_limit < 1:
        plan.warnings.append("Your plan's source limit is too small to sync this collection.")
        return plan

    current = {s.source_name: s for s in tracked_sources if not s.superseded}
    by_item: Dict[str, SourceRecord] = {}
    for s in current.values():
        if s.zotero_key and s.kind not in COLLECTION_LEVEL_KINDS:
            by_item[s.zotero_key] = s

    live_names = ({nb: {s.name for s in srcs} for nb, srcs in live_sources.items()}
                  if live_sources is not None else None)
    live_state = ({s.name: s for srcs in live_sources.values() for s in srcs}
                  if live_sources is not None else {})

    # Orphans the user has already deleted in Gemini Notebook free their slots.
    if live_names is not None:
        for nb, names in unresolved_orphans.items():
            for name in names:
                if nb in live_names and name not in live_names[nb]:
                    plan.resolved_orphans.append(name)

    # Occupied slots per existing notebook.
    occupied: Dict[str, int] = {}
    for n in existing_notebooks:
        if live_names is not None and n.notebook_name in live_names:
            occupied[n.notebook_name] = len(live_names[n.notebook_name])
        else:
            occupied[n.notebook_name] = (
                sum(1 for s in current.values() if s.notebook_name == n.notebook_name)
                + len(unresolved_orphans.get(n.notebook_name, [])))

    def present(rec: SourceRecord) -> bool:
        return live_names is None or rec.source_name in live_names.get(rec.notebook_name, set())

    # Item-level decisions.
    keys_now = {m.item.key for m in materials}
    placed: Dict[int, List[ItemMaterial]] = {}     # part → items (for bibliographies)
    to_place: List[Action] = []
    part_of_nb = {n.notebook_name: n.part for n in existing_notebooks}
    material_by_key = {m.item.key: m for m in materials}

    for m in materials:
        want = desired_source(m, cite_without_pdf)
        rec = by_item.get(m.item.key)
        if want is None:
            plan.skips.append(Skip(m.item.title, "No PDF (citation cards are turned off).", m.item.key))
            continue
        if (rec is not None and want.source_kind == URL and rec.kind == CARD
                and (rec.content_hash or "").startswith(f"fallback|{want.content_hash}|")):
            # This URL failed before and a citation card replaced it: keep the card.
            want = card_for(m, 0, None, basis_hash=want.content_hash)
            want.reason = "new"
        if rec is not None and rec.notebook_name in part_of_nb:
            part = part_of_nb[rec.notebook_name]
            live = live_state.get(rec.source_name)
            state = live.state.value if live else (rec.state or "")
            if not present(rec):
                want.part, want.reason, want.replaces = part, "missing", rec.source_name
                to_place.append(want)
                continue
            if rec.content_hash in (want.content_hash, "legacy"):
                if state == SourceState.FAILED.value:
                    code = live.error_code if live else rec.error_code
                    if rec.kind == URL and code in URL_FAILURES_THAT_FALL_BACK:
                        to_place.append(card_for(m, part, rec.source_name,
                                                 basis_hash=want.content_hash))
                    elif code in RETRYABLE_FAILURES:
                        want.part, want.reason, want.replaces = part, "retry", rec.source_name
                        to_place.append(want)
                    else:
                        plan.skips.append(Skip(m.item.title, "Gemini Notebook rejected this source: "
                                               + _reason(code, rec), m.item.key))
                    continue
                plan.unchanged += 1
                placed.setdefault(part, []).append(m)
                continue
            want.part, want.reason, want.replaces = part, "changed", rec.source_name
            to_place.append(want)
            continue
        to_place.append(want)

    # Items removed from the collection → orphans.
    for rec in current.values():
        if rec.zotero_key and rec.kind not in COLLECTION_LEVEL_KINDS and rec.zotero_key not in keys_now:
            plan.orphans.append(OrphanAction(rec.source_name, "Removed from the Zotero collection"))

    # Assign parts (sticky items keep theirs if there is room; otherwise any part with room).
    parts: Dict[int, NotebookTarget] = {n.part: NotebookTarget(n.part, n.display_name, n)
                                        for n in existing_notebooks}
    # used[p] = sources that will be live in part p (replaced ones still count: no delete).
    used: Dict[int, int] = {p: occupied[t.existing.notebook_name] for p, t in parts.items()}

    def take(preferred: int) -> int:
        if preferred and used.get(preferred, item_limit) < item_limit:
            used[preferred] += 1
            return preferred
        for p in sorted(used):
            if used[p] < item_limit:
                used[p] += 1
                return p
        p = (max(parts) if parts else 0) + 1
        parts[p] = NotebookTarget(p, "", None)
        used[p] = 1
        return p

    for a in to_place:
        a.part = take(a.part)
        plan.actions.append(a)
        placed.setdefault(a.part, []).append(material_by_key[a.zotero_key])

    # Names: "Collection" or "Collection (k/n)".
    n_parts = len(parts) or 1
    for p, t in parts.items():
        want_name = render.notebook_display_name(collection_name, p, n_parts)
        if t.existing is not None and t.existing.display_name != want_name:
            t.rename = True
        t.display_name = want_name
    plan.notebooks = [parts[p] for p in sorted(parts)]

    # Collection-level sources per part.
    for t in plan.notebooks:
        nb_name = t.existing.notebook_name if t.existing else None
        items_here = placed.get(t.part, [])
        if not items_here:
            continue
        if include_bibliography:
            title, text = render.bibliography(collection_name, [m.item for m in items_here],
                                              t.part, n_parts)
            _collection_action(plan, current, nb_name, t, BIBLIOGRAPHY, title, text, present,
                               used, cap)
        if include_annotations:
            digest = render.annotations_digest(
                collection_name, [(m.item, m.notes_html, m.annotations) for m in items_here],
                t.part, n_parts)
            if digest:
                _collection_action(plan, current, nb_name, t, ANNOTATIONS, *digest, present,
                                   used, cap)

    # Low-capacity warning: superseded versions still occupy slots.
    for t in plan.notebooks:
        if t.existing and cap - used.get(t.part, 0) <= 2:
            stuck = len(unresolved_orphans.get(t.existing.notebook_name, []))
            if stuck:
                plan.warnings.append(
                    f"“{t.display_name}” is nearly full; {stuck} outdated source(s) still take up "
                    "space. Delete them in Gemini Notebook (see the Cleanup tab) to free room.")
    return plan


def _reason(code: Optional[int], rec: SourceRecord) -> str:
    from ..notebook.base import explain_failure
    return explain_failure(code, rec.error_message or "")


def _collection_action(plan: CollectionPlan, current: Dict[str, SourceRecord], nb_name: Optional[str],
                       target: NotebookTarget, kind: str, title: str, text: str, present,
                       used: Dict[int, int], cap: int) -> None:
    h = render.content_hash(text)
    existing = None
    if nb_name:
        for s in current.values():
            if s.notebook_name == nb_name and s.kind == kind:
                existing = s
    if existing and existing.content_hash == h and present(existing):
        return
    part = target.part
    if used.get(part, 0) >= cap:
        plan.warnings.append(f"“{target.display_name}” is full, so its {kind} source wasn't "
                             "updated. Delete outdated sources (Cleanup tab) to make room.")
        return
    used[part] = used.get(part, 0) + 1
    plan.actions.append(Action(kind, part, title, "changed" if existing else "new", h,
                               text=text, replaces=existing.source_name if existing else None))
