import itertools
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from citebridge.notebook.base import Notebook, Source, SourceState
from citebridge.state import SyncState
from citebridge.zotero.client import Annotation, PdfAttachment, ZoteroItem


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CITEBRIDGE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CITEBRIDGE_OAUTH_CLIENT", raising=False)
    return tmp_path / "home"


@pytest.fixture
def state(tmp_path):
    return SyncState(str(tmp_path / "state.db"), account="me@example.com")


def make_item(key="ITEM1", title="Learning Things", last="Smith", year="2023", item_type="journalArticle",
              url="", doi="10.1/abc", version=1, abstract="An abstract.", extra_authors=()):
    creators = [{"creatorType": "author", "lastName": last, "firstName": "Jane"}]
    creators += [{"creatorType": "author", "lastName": a, "firstName": "X"} for a in extra_authors]
    return ZoteroItem(key=key, title=title, item_type=item_type, version=version, creators=creators,
                      date=f"{year}-01-01", doi=doi, url=url, abstract=abstract,
                      publication="Journal of Things")


class FakeNotebookAPI:
    """In-memory Gemini Notebook implementing NotebookWriter (no deletes, like the real API)."""

    def __init__(self, fail_urls: Optional[Dict[str, int]] = None):
        self.notebooks: Dict[str, Notebook] = {}
        self.sources: Dict[str, List[Source]] = {}
        self.uploads: List[tuple] = []
        self.fail_urls = fail_urls or {}
        self._ids = itertools.count(1)

    def _src(self, nb, display, mime="", url=None, state=SourceState.ACTIVE, code=None):
        nb = nb if nb.startswith("notebooks/") else f"notebooks/{nb}"
        s = Source(f"{nb}/sources/s{next(self._ids)}", display, state, mime, url, error_code=code)
        self.sources[nb].append(s)
        return s

    def list_notebooks(self, owned_only=True):
        return list(self.notebooks.values())

    def get_notebook(self, notebook):
        name = notebook if notebook.startswith("notebooks/") else f"notebooks/{notebook}"
        if name not in self.notebooks:
            raise KeyError(name)
        return self.notebooks[name]

    def create_notebook(self, display_name):
        nb = Notebook(f"notebooks/nb{next(self._ids)}", display_name)
        self.notebooks[nb.name] = nb
        self.sources[nb.name] = []
        return nb

    def rename_notebook(self, notebook, display_name):
        self.notebooks[notebook].display_name = display_name
        return self.notebooks[notebook]

    def list_sources(self, notebook):
        return list(self.sources[notebook])

    def get_source(self, name):
        for srcs in self.sources.values():
            for s in srcs:
                if s.name == name:
                    return s
        raise KeyError(name)

    def upload_file(self, notebook, path, display_name, mime_type=None):
        self.uploads.append((notebook, Path(path).name, display_name))
        return self._src(notebook, display_name, "application/pdf")

    def add_url(self, notebook, url, display_name=None):
        if url in self.fail_urls:
            return self._src(notebook, display_name or url, url=url, state=SourceState.FAILED,
                             code=self.fail_urls[url])
        return self._src(notebook, display_name or url, "text/html", url)

    def add_text(self, notebook, display_name, content, mime_type="text/markdown"):
        s = self._src(notebook, display_name, mime_type)
        s.text = content
        return s

    def wait_for(self, names, timeout=900, on_update=None):
        return {n: self.get_source(n) for n in names}

    def delete_in_ui(self, source_name):
        for srcs in self.sources.values():
            srcs[:] = [s for s in srcs if s.name != source_name]


class FakeZotero:
    """Stands in for ZoteroLibrary."""

    read_mode_label = "fake"

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.items: Dict[str, List[ZoteroItem]] = {}
        self.pdfs: Dict[str, PdfAttachment] = {}
        self.notes: Dict[str, List[str]] = {}
        self.annots: Dict[str, List[Annotation]] = {}
        self.created_notes: List[tuple] = []
        self.created_items: List[dict] = []

    def add(self, coll, item, pdf_md5=None, notes=(), annots=()):
        self.items.setdefault(coll, []).append(item)
        if pdf_md5 is not None:
            self.pdfs[item.key] = PdfAttachment(f"A{item.key}", f"{item.key}.pdf", md5=pdf_md5)
        self.notes[item.key] = list(notes)
        if annots:
            self.annots[f"A{item.key}"] = list(annots)

    def collection_items(self, coll):
        return list(self.items.get(coll, []))

    def children(self, key):
        return [key]

    def pdf_attachment(self, kids):
        return self.pdfs.get(kids[0])

    def notes_html(self, kids):
        return self.notes.get(kids[0], [])

    def annotations(self, att_key):
        return self.annots.get(att_key, [])

    def resolve_pdf(self, att, download_dir=None):
        p = Path(self.tmp) / att.filename
        p.write_bytes(b"%PDF-1.4 fake")
        return str(p)

    def create_note(self, parent, title, html, tags=None, collection_key=None):
        self.created_notes.append((parent, title, html, collection_key))
        return f"N{len(self.created_notes)}"

    def find_or_create_collection(self, name):
        return "COLLNEW"

    def create_item_from_source(self, **kw):
        self.created_items.append(kw)
        return f"Z{len(self.created_items)}"


@pytest.fixture
def fake_api():
    return FakeNotebookAPI()


@pytest.fixture
def fake_zot(tmp_path):
    return FakeZotero(tmp_path)
