"""
Sync engine: gather from Zotero → plan → execute against Gemini Notebook.

Execution is two-phase, unlike v1 (which blocked on every file):
  1. send every queued source, paced by the rate limiter;
  2. poll all new sources together until ACTIVE/FAILED.
Web sources that fail (paywall, unreachable, …) are automatically replaced
with a Markdown citation card so the item still reaches the notebook.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..config import AppConfig
from ..notebook.base import NotebookWriter, Source, SourceState, explain_failure
from ..state import PDF, URL, SyncState
from ..zotero.client import ZoteroLibrary
from . import planner
from .planner import Action, CollectionPlan, ItemMaterial

logger = logging.getLogger(__name__)


@dataclass
class SyncReport:
    collection_name: str
    sent: int = 0
    active: int = 0
    failed: List[str] = field(default_factory=list)
    still_processing: int = 0
    fallbacks: int = 0
    skipped: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    new_orphans: int = 0
    notebooks: List[str] = field(default_factory=list)   # notebook URLs

    @property
    def ok(self) -> bool:
        return not self.errors and not self.failed

    def summary(self) -> str:
        bits = [f"{self.sent} source(s) sent", f"{self.active} ready"]
        if self.still_processing:
            bits.append(f"{self.still_processing} still processing")
        if self.fallbacks:
            bits.append(f"{self.fallbacks} web page(s) replaced with citation cards")
        if self.failed:
            bits.append(f"{len(self.failed)} failed")
        if self.skipped:
            bits.append(f"{len(self.skipped)} skipped")
        if self.new_orphans:
            bits.append(f"{self.new_orphans} to clean up")
        return f"{self.collection_name}: " + ", ".join(bits)


class SyncEngine:
    def __init__(self, cfg: AppConfig, writer: NotebookWriter, zot: ZoteroLibrary,
                 state: SyncState, progress: Optional[Callable[[str], None]] = None):
        self.cfg = cfg
        self.writer = writer
        self.zot = zot
        self.state = state
        self.emit = progress or (lambda msg: None)

    # ── gather ──

    def gather(self, collection_key: str) -> List[ItemMaterial]:
        want_annots = self.cfg.sync.include_annotations
        out = []
        for item in self.zot.collection_items(collection_key):
            kids = self.zot.children(item.key)
            pdf = self.zot.pdf_attachment(kids)
            annots = self.zot.annotations(pdf.key) if (pdf and want_annots) else []
            out.append(ItemMaterial(item, pdf, self.zot.notes_html(kids), annots))
        return out

    def live_sources(self, collection_key: str) -> Optional[Dict[str, List[Source]]]:
        try:
            return {nb.notebook_name: self.writer.list_sources(nb.notebook_name)
                    for nb in self.state.notebooks_for(collection_key)}
        except Exception as e:
            logger.warning(f"Couldn't list live sources ({e}); planning from local state only")
            return None

    # ── plan ──

    def plan(self, collection_key: str, collection_name: str) -> CollectionPlan:
        self.emit(f"Reading “{collection_name}” from {self.zot.read_mode_label or 'Zotero'}…")
        materials = self.gather(collection_key)
        notebooks = self.state.notebooks_for(collection_key)
        orphans: Dict[str, List[str]] = {}
        for o in self.state.orphans():
            orphans.setdefault(o.notebook_name, []).append(o.source_name)
        return planner.plan_collection(
            collection_key, collection_name, materials,
            existing_notebooks=notebooks,
            tracked_sources=self.state.sources_for_collection(collection_key, include_superseded=False),
            unresolved_orphans=orphans,
            live_sources=self.live_sources(collection_key),
            cap=self.cfg.sync.sources_per_notebook,
            include_bibliography=self.cfg.sync.include_bibliography,
            include_annotations=self.cfg.sync.include_annotations,
            cite_without_pdf=self.cfg.sync.cite_items_without_pdf)

    # ── execute ──

    def execute(self, plan: CollectionPlan, wait: bool = True,
                timeout: float = 900) -> SyncReport:
        rep = SyncReport(plan.collection_name, skipped=[f"{s.title}: {s.reason}" for s in plan.skips])
        for name in plan.resolved_orphans:
            self.state.resolve_orphan(name)

        # 1. notebooks
        nb_for_part: Dict[int, str] = {}
        for t in plan.notebooks:
            try:
                if t.existing is None:
                    # Re-running a plan (e.g. after an interrupted Send) must not create a
                    # second notebook for a part that already has one.
                    already = next((n for n in self.state.notebooks_for(plan.collection_key)
                                    if n.part == t.part), None)
                    if already is not None:
                        t.existing = already
                if t.existing is None:
                    self.emit(f"Creating notebook “{t.display_name}”…")
                    nb = self.writer.create_notebook(t.display_name)
                    nb_for_part[t.part] = nb.name
                else:
                    nb_for_part[t.part] = t.existing.notebook_name
                    if t.rename:
                        self.writer.rename_notebook(t.existing.notebook_name, t.display_name)
                self.state.upsert_notebook(plan.collection_key, t.part, nb_for_part[t.part],
                                           t.display_name)
                rep.notebooks.append(f"https://notebook.google.com/notebook/"
                                     f"{nb_for_part[t.part].rsplit('/', 1)[-1]}")
            except Exception as e:
                rep.errors.append(f"Notebook “{t.display_name}”: {e}")
        if rep.errors:
            self.state.log("sync", "error", rep.summary())
            return rep

        # 2. orphans (items removed from the collection)
        for o in plan.orphans:
            self.state.supersede(o.source_name, o.reason)
            rep.new_orphans += 1

        # 3. sources — items first, collection-level (bibliography) last
        ordered = sorted(plan.actions, key=lambda a: a.zotero_key is None)
        download_dir = tempfile.mkdtemp(prefix="citebridge_")
        sent: Dict[str, Action] = {}
        for i, a in enumerate(ordered, 1):
            nb = nb_for_part[a.part]
            if self._already_sent(plan.collection_key, nb, a):
                continue
            self.emit(f"[{i}/{len(ordered)}] {a.display_name}")
            try:
                src = self._send(a, nb, download_dir)
            except Exception as e:
                rep.errors.append(f"{a.display_name}: {e}")
                continue
            if src is None:
                rep.skipped.append(f"{a.display_name}: PDF file not available locally or via Zotero")
                continue
            self._record(plan.collection_key, nb, a, src)
            sent[src.name] = a
            rep.sent += 1

        # 4. poll processing
        if wait and sent:
            self.emit(f"Waiting for Gemini Notebook to process {len(sent)} source(s)…")
            try:
                done = self.writer.wait_for(list(sent), timeout=timeout,
                                            on_update=lambda s: self.emit(
                                                f"  {s.state.value.lower()}: {s.display_name}"))
            except Exception as e:
                rep.errors.append(f"Checking processing status failed: {e} (see the Processing tab)")
                done = {}
            for name, a in sent.items():
                src = done.get(name)
                if src is None:
                    rep.still_processing += 1
                    continue
                self.state.update_source_state(name, src.state.value, src.error_code, src.error_message)
                if src.state == SourceState.ACTIVE:
                    rep.active += 1
                elif a.source_kind == URL and src.error_code in planner.URL_FAILURES_THAT_FALL_BACK:
                    if self._fallback_to_card(plan, a, src, nb_for_part[a.part]):
                        rep.fallbacks += 1
                else:
                    rep.failed.append(f"{a.display_name}: {explain_failure(src.error_code, src.error_message)}")
        else:
            rep.still_processing = len(sent)

        for nb in nb_for_part.values():
            self.state.touch_notebook(nb)
        self.state.log("sync", "success" if rep.ok else "partial", rep.summary())
        self.emit(rep.summary())
        return rep

    def _send(self, a: Action, notebook: str, download_dir: str) -> Optional[Source]:
        if a.source_kind == PDF:
            path = self.zot.resolve_pdf(a.pdf, download_dir)
            if not path:
                return None
            return self.writer.upload_file(notebook, path, a.display_name, "application/pdf")
        if a.source_kind == URL:
            return self.writer.add_url(notebook, a.url, a.display_name)
        return self.writer.add_text(notebook, a.display_name, a.text or "")

    def _already_sent(self, collection_key: str, notebook: str, a: Action) -> bool:
        rec = (self.state.current_source_for_item(collection_key, a.zotero_key) if a.zotero_key
               else self.state.collection_source(collection_key, notebook, a.source_kind))
        return bool(rec and rec.content_hash == a.content_hash and rec.source_name != a.replaces)

    def _record(self, collection_key: str, notebook: str, a: Action, src: Source) -> None:
        if a.replaces:
            if a.reason == "missing":          # already gone from the notebook
                self.state.supersede(a.replaces, "Deleted in Gemini Notebook")
                self.state.resolve_orphan(a.replaces)
            else:
                self.state.supersede(a.replaces, {
                    "changed": "Replaced by an updated version",
                    "fallback": "Web page failed; replaced with a citation card",
                    "retry": "Failed earlier; sent again",
                }.get(a.reason, "Replaced"))
        self.state.record_source(
            collection_key=collection_key, notebook_name=notebook, source_name=src.name,
            kind=a.source_kind, display_name=a.display_name, zotero_key=a.zotero_key,
            content_hash=a.content_hash, zotero_version=a.zotero_version,
            state=src.state.value)

    def _fallback_to_card(self, plan: CollectionPlan, a: Action, failed: Source, notebook: str) -> bool:
        m = plan.materials.get(a.zotero_key or "")
        if m is None:
            return False
        card = planner.card_for(m, a.part, failed.name, basis_hash=a.content_hash)
        try:
            src = self.writer.add_text(notebook, card.display_name, card.text or "")
        except Exception as e:
            logger.error(f"Citation-card fallback failed for {a.display_name}: {e}")
            return False
        self._record(plan.collection_key, notebook, card, src)
        return True

    # ── status ──

    def refresh_pending(self) -> int:
        """Re-check sources still PROCESSING from earlier runs; returns how many changed."""
        changed = 0
        for rec in self.state.pending_sources():
            try:
                src = self.writer.get_source(rec.source_name)
            except Exception as e:
                logger.warning(f"Status check failed for {rec.display_name}: {e}")
                continue
            if src.state.value != (rec.state or ""):
                self.state.update_source_state(rec.source_name, src.state.value,
                                               src.error_code, src.error_message)
                changed += 1
        return changed

    # ── v1 → v2 adoption ──

    def adopt_legacy(self) -> List[str]:
        """Link notebooks created by CiteBridge v1 so they aren't duplicated."""
        adopted = []
        legacy = self.state.legacy_collections()
        if not legacy:
            return adopted
        try:
            by_title = {}
            for nb in self.writer.list_notebooks():
                by_title.setdefault(nb.display_name, nb)
        except Exception as e:
            logger.warning(f"Legacy adoption skipped: {e}")
            return adopted
        for lc in legacy:
            nb = None
            try:
                nb = self.writer.get_notebook(lc.notebook_id)
            except Exception:
                nb = by_title.get(lc.collection_name)
            if nb is None:
                continue
            self.state.upsert_notebook(lc.collection_key, 1, nb.name, nb.display_name)
            # v1 named sources after the Zotero title; register what we can match so
            # those items aren't uploaded twice.
            try:
                from collections import Counter
                zitems = self.zot.collection_items(lc.collection_key)
                srcs = self.writer.list_sources(nb.name)
                norm = lambda t: (t or "").strip().lower()  # noqa: E731
                item_titles = Counter(norm(i.title) for i in zitems)
                src_titles = Counter(norm(s.display_name) for s in srcs)
                # Only map titles that are unique on BOTH sides; ambiguous ones stay
                # unmapped (worst case: one re-upload, never a wrong attribution).
                items = {norm(i.title): i for i in zitems if item_titles[norm(i.title)] == 1}
                for src in srcs:
                    if src_titles[norm(src.display_name)] != 1:
                        continue
                    item = items.get(norm(src.display_name))
                    if item:
                        self.state.record_source(
                            collection_key=lc.collection_key, notebook_name=nb.name,
                            source_name=src.name, kind=PDF, display_name=src.display_name,
                            zotero_key=item.key, content_hash="legacy",
                            zotero_version=item.version, state=src.state.value)
            except Exception as e:
                logger.warning(f"Couldn't map legacy sources for {lc.collection_name}: {e}")
            self.state.forget_legacy_collection(lc.collection_key)
            adopted.append(lc.collection_name)
        return adopted
