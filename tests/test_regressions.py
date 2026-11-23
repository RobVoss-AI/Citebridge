"""Regression tests for the independent correctness review's findings."""

import httpx
import pytest

from citebridge.config import AppConfig
from citebridge.notebook.base import Note, Notebook, SourceState
from citebridge.notebook.official import ApiError, OfficialNotebookClient
from citebridge.ratelimit import ApiLimits
from citebridge.state import CARD
from citebridge.sync import reverse
from citebridge.sync.executor import SyncEngine

from .conftest import FakeNotebookAPI, make_item


def eng(api, zot, state, **sync):
    cfg = AppConfig()
    for k, v in sync.items():
        setattr(cfg.sync, k, v)
    return SyncEngine(cfg, api, zot, state)


def sync(e, key="C1", name="Coll"):
    p = e.plan(key, name)
    return p, e.execute(p)


def live_counts(api):
    return {nb: len(srcs) for nb, srcs in api.sources.items()}


# 1 ─ a failed web page must not be retried as a URL on every sync
def test_url_fallback_is_stable_across_syncs(fake_zot, state):
    api = FakeNotebookAPI(fail_urls={"https://example.org/post": 7})
    fake_zot.add("C1", make_item("W1", "Post", "Lee", item_type="blogPost", url="https://example.org/post"))
    e = eng(api, fake_zot, state, include_bibliography=False, include_annotations=False)
    sync(e)
    for _ in range(3):
        plan, rep = sync(e)
        assert plan.is_noop and rep.sent == 0
    assert sum(live_counts(api).values()) == 2          # failed URL + its citation card, no churn
    assert len(state.orphans()) == 1


def test_fallback_card_still_updates_when_its_content_changes(fake_zot, state):
    api = FakeNotebookAPI(fail_urls={"https://example.org/post": 7})
    item = make_item("W1", "Post", "Lee", item_type="blogPost", url="https://example.org/post")
    fake_zot.add("C1", item)
    e = eng(api, fake_zot, state, include_bibliography=False, include_annotations=False)
    sync(e)
    item.abstract = "A new abstract"
    plan, _ = sync(e)
    assert [(a.source_kind, a.reason) for a in plan.actions] == [(CARD, "changed")]


# 2 ─ no notebook is ever planned past its cap, however the collection grows
@pytest.mark.parametrize("annotations", [False, True])
def test_live_sources_never_exceed_cap(fake_zot, state, annotations):
    from citebridge.zotero.client import Annotation
    api, cap = FakeNotebookAPI(), 5
    e = eng(api, fake_zot, state, include_annotations=annotations, source_cap_override=cap)
    for n in range(12):
        fake_zot.add("C1", make_item(f"K{n}", f"Paper {n}", f"Auth{n}"), pdf_md5=str(n),
                     annots=[Annotation(f"quote {n}")])
        sync(e)
        assert max(live_counts(api).values()) <= cap, live_counts(api)


def test_turning_bibliography_off_does_not_overcount_space(fake_zot, state):
    api = FakeNotebookAPI()
    for n in range(3):
        fake_zot.add("C1", make_item(f"K{n}", f"P{n}", f"A{n}"), pdf_md5=str(n))
    e = eng(api, fake_zot, state, include_annotations=False, source_cap_override=5)
    sync(e)
    e.cfg.sync.include_bibliography = False
    for n in range(3, 6):
        fake_zot.add("C1", make_item(f"K{n}", f"P{n}", f"A{n}"), pdf_md5=str(n))
    sync(e)
    assert max(live_counts(api).values()) <= 5


# 3 ─ re-executing a plan (interrupted Send) creates no duplicates
def test_executing_the_same_plan_twice_is_idempotent(fake_api, fake_zot, state):
    fake_zot.add("C1", make_item("P1", "Paper", "Smith"), pdf_md5="a")
    e = eng(fake_api, fake_zot, state)
    plan = e.plan("C1", "Coll")
    e.execute(plan)
    rep = e.execute(plan)
    assert len(fake_api.notebooks) == 1 and rep.sent == 0
    assert len(fake_api.uploads) == 1


# 4 ─ notes are only marked imported once Zotero has them
def test_failed_note_writes_are_retried_not_lost(fake_zot, state):
    fake_zot.add("C1", make_item("L1", last="Lee"))
    state.upsert_notebook("C1", 1, "notebooks/nb1", "Coll")

    class Reader:
        def list_notes(self, nb):
            return [Note("n1", "On Lee", "Lee (2023) shows"), Note("n2", "General", "themes")]

    real = fake_zot.create_note
    fake_zot.create_note = lambda *a, **k: None                  # pyzotero 'failed' without raising
    rep = reverse.pull_notes("C1", "Coll", Reader(), fake_zot, state)
    assert rep.attached == 0 and rep.errors
    assert not state.note_imported("n1") and not state.note_imported("n2")

    def boom(*a, **k):
        raise RuntimeError("network")
    fake_zot.create_note = lambda parent, *a, **k: real(parent, *a, **k) if parent else boom()
    reverse.pull_notes("C1", "Coll", Reader(), fake_zot, state)
    assert state.note_imported("n1") and not state.note_imported("n2")   # leftover kept for retry

    fake_zot.create_note = real
    rep = reverse.pull_notes("C1", "Coll", Reader(), fake_zot, state)
    assert rep.to_collection_note == 1 and state.note_imported("n2")


# 5 ─ label matching respects word and co-author boundaries
def test_attribution_needs_whole_labels():
    idx = reverse._label_index([make_item("S", last="Smith"), make_item("J", last="Jones")])
    assert reverse.attribute_note("As Goldsmith (2023) argues", idx) is None
    assert reverse.attribute_note("Smith & Jones (2023) found", idx) is None
    assert reverse.attribute_note("Smith and Jones (2023) found", idx) is None
    assert reverse.attribute_note("Per Jones (2023), yes", idx) == "J"


# 6 ─ legacy adoption never maps ambiguous titles
def test_legacy_adoption_skips_shared_titles(tmp_path, fake_api, fake_zot):
    import sqlite3
    from citebridge.state import SyncState
    db = tmp_path / "v1.db"
    with sqlite3.connect(db) as c:
        c.executescript("""CREATE TABLE synced_collections (id INTEGER PRIMARY KEY, zotero_key TEXT UNIQUE,
            zotero_name TEXT, nlm_notebook_id TEXT, last_synced TEXT);
            INSERT INTO synced_collections (zotero_key, zotero_name, nlm_notebook_id) VALUES ('C1','Coll','nb1');""")
    st = SyncState(str(db), account="me")
    fake_api.notebooks["notebooks/nb1"] = Notebook("notebooks/nb1", "Coll")
    fake_api.sources["notebooks/nb1"] = []
    fake_api._src("notebooks/nb1", "Introduction", "application/pdf")
    fake_zot.add("C1", make_item("K1", "Introduction", "Abe"), pdf_md5="1")
    fake_zot.add("C1", make_item("K2", "Introduction", "Bo"), pdf_md5="2")
    SyncEngine(AppConfig(), fake_api, fake_zot, st).adopt_legacy()
    assert st.item_key_for_source("notebooks/nb1/sources/s1") is None    # not guessed


# 7 ─ capacity / internal failures are retried
def test_source_limit_failure_is_retried(fake_api, fake_zot, state):
    fake_zot.add("C1", make_item("P1", "Paper", "Smith"), pdf_md5="a")
    e = eng(fake_api, fake_zot, state, include_bibliography=False, include_annotations=False)
    orig = fake_api.upload_file

    def full(nb, path, name, mime=None):
        s = orig(nb, path, name, mime)
        s.state, s.error_code = SourceState.FAILED, 8
        return s
    fake_api.upload_file = full
    sync(e)
    fake_api.upload_file = orig
    plan, rep = sync(e)
    assert [a.reason for a in plan.actions] == ["retry"] and rep.active == 1


# 8 ─ editing a parent item's metadata doesn't re-upload a PDF without md5
def test_metadata_edit_does_not_reupload_pdf_without_md5(fake_api, fake_zot, state):
    item = make_item("P1", "Paper", "Smith")
    fake_zot.add("C1", item, pdf_md5="")
    e = eng(fake_api, fake_zot, state, include_bibliography=False, include_annotations=False)
    sync(e)
    item.version += 5                                    # tag added in Zotero
    plan, _ = sync(e)
    assert plan.is_noop


# 9 ─ untracked copies of our own sources are never imported back
def test_import_skips_copies_of_citebridge_sources(fake_api, fake_zot, state):
    fake_zot.add("C1", make_item("P1", "Paper", "Smith"), pdf_md5="a")
    sync(eng(fake_api, fake_zot, state, include_bibliography=False, include_annotations=False))
    nb = next(iter(fake_api.notebooks))
    fake_api._src(nb, "Smith (2023) — Paper", "application/pdf")       # duplicate after a 5xx
    fake_api._src(nb, "Someone's blog", "text/html", "https://blog.example")
    rep = reverse.import_new_sources(nb, "Coll", fake_api, fake_zot, state)
    assert rep.imported == 1 and fake_zot.created_items[0]["title"] == "Someone's blog"


# residual risks ─ no blind retry of committed creates; polling survives deletes
def _client(handler):
    return OfficialNotebookClient(lambda force=False: "t",
                                  http=httpx.Client(transport=httpx.MockTransport(handler)),
                                  limits=ApiLimits(sleep=lambda s: None), sleep=lambda s: None)


def test_create_is_not_retried_after_5xx():
    calls = []

    def h(req):
        calls.append(req)
        return httpx.Response(503, json={"error": {"message": "backend"}})
    with pytest.raises(ApiError, match="may or may not"):
        _client(h).add_text("nb", "x", "y")
    assert len(calls) == 1


def test_reads_are_still_retried_after_5xx():
    calls = []

    def h(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(503, json={"error": {"message": "backend"}})
        return httpx.Response(200, json={"name": "notebooks/a"})
    _client(h).get_notebook("a")
    assert len(calls) == 2


def test_wait_for_survives_a_deleted_source():
    def h(req):
        if req.url.path.endswith("/gone"):
            return httpx.Response(404, json={"error": {"message": "not found"}})
        return httpx.Response(200, json={"name": "notebooks/n/sources/ok", "state": "ACTIVE"})
    done = _client(h).wait_for(["notebooks/n/sources/gone", "notebooks/n/sources/ok"])
    assert list(done) == ["notebooks/n/sources/ok"]
