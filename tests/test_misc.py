"""State migration, rendering, note attribution, config and token storage."""

import os
import sqlite3
import stat

from citebridge.config import AppConfig, config_file
from citebridge.notebook.auth import FileTokenStore
from citebridge.state import SyncState
from citebridge.sync import reverse
from citebridge.sync.executor import SyncEngine
from citebridge.zotero import render
from citebridge.zotero.client import Annotation, parse_item

from .conftest import make_item


# ── state migration ──

V1_SCHEMA = """
CREATE TABLE synced_collections (id INTEGER PRIMARY KEY AUTOINCREMENT, zotero_key TEXT UNIQUE NOT NULL,
  zotero_name TEXT NOT NULL, nlm_notebook_id TEXT, last_synced TIMESTAMP);
CREATE TABLE synced_items (id INTEGER PRIMARY KEY AUTOINCREMENT, zotero_key TEXT NOT NULL,
  collection_zotero_key TEXT NOT NULL, title TEXT, file_hash TEXT, nlm_source_id TEXT, last_synced TIMESTAMP,
  UNIQUE(zotero_key, collection_zotero_key));
CREATE TABLE nlm_notes_synced (id INTEGER PRIMARY KEY AUTOINCREMENT, nlm_notebook_id TEXT NOT NULL,
  nlm_note_id TEXT NOT NULL, zotero_item_key TEXT, zotero_note_key TEXT, synced_at TIMESTAMP, UNIQUE(nlm_note_id));
CREATE TABLE sync_log (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  action TEXT NOT NULL, status TEXT NOT NULL, details TEXT);
INSERT INTO synced_collections (zotero_key, zotero_name, nlm_notebook_id) VALUES ('C1', 'AI Ethics', 'nb1');
INSERT INTO nlm_notes_synced (nlm_notebook_id, nlm_note_id) VALUES ('nb1', 'note-1');
INSERT INTO sync_log (action, status, details) VALUES ('full_sync', 'success', 'old run');
"""


def test_v1_database_migrates_and_its_notebooks_are_adopted(tmp_path, fake_api, fake_zot):
    db = tmp_path / "v1.db"
    with sqlite3.connect(db) as c:
        c.executescript(V1_SCHEMA)
    st = SyncState(str(db), account="me@example.com")
    assert [lc.notebook_id for lc in st.legacy_collections()] == ["nb1"]
    assert st.note_imported("note-1")                          # v1 imports aren't redone
    assert any("old run" in (r[3] or "") for r in st.recent_logs())

    # v1 notebook exists in Gemini Notebook with a source v1 named after the Zotero title
    from citebridge.notebook.base import Notebook
    fake_api.notebooks["notebooks/nb1"] = Notebook("notebooks/nb1", "AI Ethics")
    fake_api.sources["notebooks/nb1"] = []
    fake_api._src("notebooks/nb1", "Fairness in ML", "application/pdf")
    fake_zot.add("C1", make_item("P1", "Fairness in ML", "Smith"), pdf_md5="aaa")

    eng = SyncEngine(AppConfig(), fake_api, fake_zot, st)
    assert eng.adopt_legacy() == ["AI Ethics"]
    assert st.legacy_collections() == []
    plan = eng.plan("C1", "AI Ethics")
    assert plan.notebooks[0].existing is not None             # no duplicate notebook
    assert all(a.zotero_key != "P1" for a in plan.actions)    # no duplicate upload

    SyncState(str(db), account="me@example.com")              # re-opening is idempotent


def test_state_is_scoped_per_account(tmp_path):
    db = str(tmp_path / "s.db")
    a, b = SyncState(db, "a@x.com"), SyncState(db, "b@x.com")
    a.upsert_notebook("C1", 1, "notebooks/n1", "AI")
    assert len(a.notebooks_for("C1")) == 1 and b.notebooks_for("C1") == []


# ── rendering ──

def test_author_year_labels():
    assert render.author_year(make_item(last="Smith")) == "Smith (2023)"
    assert render.author_year(make_item(last="Smith", extra_authors=["Lee"])) == "Smith & Lee (2023)"
    assert render.author_year(make_item(last="Smith", extra_authors=["Lee", "Ng"])) == "Smith et al. (2023)"
    no_date = make_item()
    no_date.date = ""
    assert render.author_year(no_date) == "Smith (n.d.)"
    assert len(render.display_name(make_item(title="x" * 900))) == 500


def test_citation_card_carries_metadata_notes_and_highlights():
    card = render.citation_card(make_item(), ["<p>My <b>key</b> note</p>"],
                                [Annotation("a quote", "why it matters", "12")])
    for needle in ("# Learning Things", "Smith, J. (2023)", "https://doi.org/10.1/abc",
                   "`ITEM1`", "## Abstract", "My key note", "“a quote” (p. 12)", "why it matters",
                   "cite this item as **Smith (2023)**"):
        assert needle in card, needle


def test_bibliography_lists_every_item_with_its_key():
    title, md = render.bibliography("AI", [make_item("K2", last="Zed"), make_item("K1", last="Abe")])
    assert title == "📚 Bibliography — AI"
    assert md.index("Abe (2023)") < md.index("Zed (2023)")    # sorted
    assert "`K1`" in md and "`K2`" in md and "2 items." in md


def test_text_is_kept_under_the_1mb_limit():
    out = render.fit("é" * 900_000)
    assert len(out.encode("utf-8")) <= 1_000_000 and "truncated" in out


def test_parse_item_reads_better_bibtex_citation_key():
    item = parse_item({"data": {"key": "K", "itemType": "book", "title": "T", "version": 7,
                                "extra": "tex.foo: 1\nCitation Key: smith2023things"}})
    assert (item.citation_key, item.version) == ("smith2023things", 7)


# ── reverse attribution ──

def test_note_attribution_is_never_guessed():
    items = [make_item("S1", last="Smith"), make_item("L1", last="Lee"),
             make_item("S2", last="Smith", title="Other")]           # ambiguous label
    items[2].date = "2023"
    idx = reverse._label_index(items + [make_item("A1", last="", title="Anon")])
    idx_clean = reverse._label_index(items[:2])
    assert reverse.attribute_note("Lee (2023) argues X", idx_clean) == "L1"
    assert reverse.attribute_note("Lee (2023) vs Smith (2023)", idx_clean) is None   # several
    assert reverse.attribute_note("no citation here", idx_clean) is None
    assert reverse.attribute_note("Smith (2023) says", idx) is None   # label shared by 2 items
    assert "(2023)" not in idx                                        # author-less labels excluded


def test_pull_notes_attaches_or_collects(state, fake_zot):
    from citebridge.notebook.base import Note
    fake_zot.add("C1", make_item("L1", last="Lee"))
    state.upsert_notebook("C1", 1, "notebooks/nb1", "AI")

    class Reader:
        def list_notes(self, nb):
            return [Note("n1", "Summary", "Lee (2023) shows..."), Note("n2", "Overview", "Broad themes")]

    rep = reverse.pull_notes("C1", "AI", Reader(), fake_zot, state)
    assert (rep.attached, rep.to_collection_note) == (1, 1)
    parents = [n[0] for n in fake_zot.created_notes]
    assert "L1" in parents and None in parents                       # one item note, one collection note
    assert reverse.pull_notes("C1", "AI", Reader(), fake_zot, state).skipped == 2  # idempotent


# ── config & tokens ──

def test_v1_config_loads_without_errors(isolated_home):
    config_file().parent.mkdir(parents=True, exist_ok=True)
    config_file().write_text(
        "zotero: {api_key: K, library_id: '1', library_type: user, local_storage_path: ''}\n"
        "notebooklm: {storage_path: /x/state.json, authenticated: true}\n"
        "sync: {enabled_collections: [C1], sync_notes_back: true, max_file_size_mb: 200,"
        " auto_sync_interval_minutes: 0}\n")
    cfg = AppConfig.load()
    assert cfg.zotero.api_key == "K" and cfg.sync.enabled_collections == ["C1"]
    assert cfg.experimental.storage_path == "/x/state.json"
    cfg.save()
    assert stat.S_IMODE(os.stat(config_file()).st_mode) == 0o600


def test_token_store_is_private_and_per_account(tmp_path):
    store = FileTokenStore(tmp_path / "accounts")
    store.save("me@example.com", '{"refresh_token": "r"}')
    store.save("other@example.com", '{"refresh_token": "o"}')
    assert store.accounts() == ["me@example.com", "other@example.com"]
    p = tmp_path / "accounts" / "me@example.com" / "token.json"
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    store.delete("me@example.com")
    assert store.load("me@example.com") is None
