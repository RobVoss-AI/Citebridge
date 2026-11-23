"""End-to-end engine scenarios against the in-memory fake API and Zotero."""

from citebridge.config import AppConfig
from citebridge.state import BIBLIOGRAPHY, CARD, PDF, URL
from citebridge.sync.executor import SyncEngine
from citebridge.zotero.client import Annotation

from .conftest import FakeNotebookAPI, make_item


def engine(fake_api, fake_zot, state, **sync):
    cfg = AppConfig()
    for k, v in sync.items():
        setattr(cfg.sync, k, v)
    return SyncEngine(cfg, fake_api, fake_zot, state)


def run(eng, key="C1", name="AI Ethics"):
    plan = eng.plan(key, name)
    return plan, eng.execute(plan)


def seed(fake_zot):
    fake_zot.add("C1", make_item("P1", "Fairness in ML", "Smith"), pdf_md5="aaa",
                 annots=[Annotation("bias is structural", "key claim", "4")])
    fake_zot.add("C1", make_item("W1", "A blog post", "Lee", item_type="blogPost",
                                 url="https://example.org/post", doi=""))
    fake_zot.add("C1", make_item("A1", "Abstract only", "Ng", doi="10.9/xyz"),
                 notes=["<p>My reading note</p>"])


def test_first_sync_sends_every_item_plus_collection_sources(fake_api, fake_zot, state):
    seed(fake_zot)
    plan, rep = run(engine(fake_api, fake_zot, state))
    kinds = sorted(a.source_kind for a in plan.actions)
    assert kinds == sorted([PDF, URL, CARD, BIBLIOGRAPHY, "annotations"])
    assert rep.ok and rep.sent == 5 and rep.active == 5
    nb = next(iter(fake_api.notebooks.values()))
    assert nb.display_name == "AI Ethics"
    names = [s.display_name for s in fake_api.sources[nb.name]]
    assert "Smith (2023) — Fairness in ML" in names          # author-year display names
    # collection-level sources (bibliography, highlights) are sent after every item
    assert {n[0] for n in names[-2:]} == {"📚", "🖍"}
    # every item source maps back to its Zotero key
    for rec in state.sources_for_collection("C1"):
        if rec.kind in (PDF, URL, CARD):
            assert state.item_key_for_source(rec.source_name) == rec.zotero_key


def test_resync_is_a_noop(fake_api, fake_zot, state):
    seed(fake_zot)
    eng = engine(fake_api, fake_zot, state)
    run(eng)
    plan, rep = run(eng)
    assert plan.is_noop and plan.unchanged == 3 and rep.sent == 0


def test_changed_pdf_replaces_source_and_lists_orphan(fake_api, fake_zot, state):
    seed(fake_zot)
    eng = engine(fake_api, fake_zot, state, include_annotations=False)
    run(eng)
    old = state.current_source_for_item("C1", "P1").source_name
    fake_zot.pdfs["P1"].md5 = "bbb"                            # PDF edited in Zotero
    plan, rep = run(eng)
    assert [a.reason for a in plan.actions if a.zotero_key == "P1"] == ["changed"]
    assert state.current_source_for_item("C1", "P1").source_name != old
    assert [o.source_name for o in state.orphans()] == [old]   # can't delete → cleanup list


def test_removed_item_becomes_orphan(fake_api, fake_zot, state):
    seed(fake_zot)
    eng = engine(fake_api, fake_zot, state, include_bibliography=False, include_annotations=False)
    run(eng)
    fake_zot.items["C1"] = [i for i in fake_zot.items["C1"] if i.key != "W1"]
    plan, _ = run(eng)
    assert [o.reason for o in plan.orphans] == ["Removed from the Zotero collection"]
    assert len(state.orphans()) == 1


def test_paywalled_url_falls_back_to_citation_card(fake_zot, state):
    api = FakeNotebookAPI(fail_urls={"https://example.org/post": 7})
    fake_zot.add("C1", make_item("W1", "A blog post", "Lee", item_type="blogPost",
                                 url="https://example.org/post"))
    eng = engine(api, fake_zot, state, include_bibliography=False, include_annotations=False)
    _, rep = run(eng)
    assert rep.fallbacks == 1 and not rep.failed
    assert state.current_source_for_item("C1", "W1").kind == CARD
    assert state.orphans()[0].reason.startswith("Web page failed")


def test_source_deleted_in_web_ui_is_re_added_and_orphan_auto_resolves(fake_api, fake_zot, state):
    seed(fake_zot)
    eng = engine(fake_api, fake_zot, state, include_bibliography=False, include_annotations=False)
    run(eng)
    rec = state.current_source_for_item("C1", "A1")
    fake_api.delete_in_ui(rec.source_name)
    plan, _ = run(eng)
    assert [a.reason for a in plan.actions] == ["missing"]
    assert state.orphans() == []                               # nothing for the user to clean up


def test_collection_over_cap_splits_into_parts(fake_api, fake_zot, state):
    for n in range(7):
        fake_zot.add("C1", make_item(f"K{n}", f"Paper {n}", f"Author{n}"), pdf_md5=str(n))
    # cap 5, minus 1 reserved for the bibliography → 4 items per notebook
    eng = engine(fake_api, fake_zot, state, include_annotations=False)
    from citebridge.sync import planner as P
    plan = P.plan_collection("C1", "Big", eng.gather("C1"), existing_notebooks=[],
                             tracked_sources=[], unresolved_orphans={}, live_sources={},
                             cap=5, include_annotations=False)
    assert [n.display_name for n in plan.notebooks] == ["Big (1/2)", "Big (2/2)"]
    assert plan.count(PDF) == 7 and plan.count(BIBLIOGRAPHY) == 2


def test_growing_collection_renames_existing_notebook_when_it_splits(fake_api, fake_zot, state):
    for n in range(3):
        fake_zot.add("C1", make_item(f"K{n}", f"Paper {n}", f"Author{n}"), pdf_md5=str(n))
    eng = engine(fake_api, fake_zot, state, include_annotations=False, source_cap_override=4)
    run(eng, name="Grow")
    for n in range(3, 6):
        fake_zot.add("C1", make_item(f"K{n}", f"Paper {n}", f"Author{n}"), pdf_md5=str(n))
    plan, rep = run(eng, name="Grow")
    assert rep.ok
    assert sorted(nb.display_name for nb in fake_api.notebooks.values()) == ["Grow (1/2)", "Grow (2/2)"]
    assert any(t.rename for t in plan.notebooks)


def test_items_without_pdf_skipped_when_cards_disabled(fake_api, fake_zot, state):
    seed(fake_zot)
    plan = engine(fake_api, fake_zot, state, cite_items_without_pdf=False).plan("C1", "AI Ethics")
    assert {s.zotero_key for s in plan.skips} == {"W1", "A1"}
