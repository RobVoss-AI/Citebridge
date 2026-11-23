"""
CiteBridge v2 — Zotero ↔ Gemini Notebook (formerly NotebookLM).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))  # however the app is launched

from citebridge import __version__  # noqa: E402
from citebridge.config import TIER_LIMITS, AppConfig, citebridge_home  # noqa: E402
from citebridge.notebook import auth  # noqa: E402
from citebridge.notebook.experimental import ExperimentalReader  # noqa: E402
from citebridge.notebook.official import ApiError, OfficialNotebookClient  # noqa: E402
from citebridge.state import SyncState  # noqa: E402
from citebridge.sync import reverse  # noqa: E402
from citebridge.sync.executor import SyncEngine  # noqa: E402
from citebridge.zotero.client import ZoteroLibrary, ZoteroUnavailable, detect_storage_path  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

st.set_page_config(page_title="CiteBridge", page_icon="📚", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&display=swap');
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; font-weight: 400 !important; }
.cb-title { font-family: 'DM Serif Display', serif; font-size: 2.4rem; margin: 0; color: #1a1a2e; }
.cb-title span { color: #c0392b; }
.cb-sub { color: #8a8578; margin-top: -4px; }
</style>""", unsafe_allow_html=True)

ss = st.session_state
if "cfg" not in ss:
    ss.cfg = AppConfig.load()
cfg: AppConfig = ss.cfg


def save():
    cfg.save()


def friendly(e: Exception) -> str:
    return str(e) if isinstance(e, (ApiError, auth.AuthError, ZoteroUnavailable)) else f"{type(e).__name__}: {e}"


# ── objects for the current account ──

account = cfg.google.account if cfg.google.account in auth.signed_in_accounts() else ""
state = SyncState(account=account)
zot = ZoteroLibrary(cfg.zotero)


def writer() -> OfficialNotebookClient:
    return OfficialNotebookClient(auth.token_provider(account), quota_project=cfg.google.quota_project)


def engine(progress=None) -> SyncEngine:
    return SyncEngine(cfg, writer(), zot, state, progress)


def reader():
    if not cfg.experimental.enabled:
        return None
    ok, _ = ExperimentalReader.available(cfg.experimental.storage_path)
    return ExperimentalReader(cfg.experimental.storage_path) if ok else None


# ══════════════════ SIDEBAR ══════════════════

with st.sidebar:
    st.markdown("### Google account")
    if not auth.has_client_config():
        st.warning("CiteBridge's Google sign-in file is missing. Put `oauth_client.json` in "
                   f"`{citebridge_home()}` (it comes with your CiteBridge download).")
    else:
        accounts = auth.signed_in_accounts()
        if accounts:
            choice = st.selectbox("Signed in as", accounts,
                                  index=accounts.index(account) if account in accounts else 0)
            if choice != cfg.google.account:
                cfg.google.account = choice
                save()
                st.rerun()
        c1, c2 = st.columns(2)
        if c1.button("Sign in with Google", use_container_width=True,
                     type="primary" if not accounts else "secondary"):
            with st.spinner("Finish signing in in the browser tab that just opened…"):
                try:
                    cfg.google.account = auth.sign_in()
                    save()
                    st.rerun()
                except Exception as e:
                    st.error(friendly(e))
        if accounts and c2.button("Sign out", use_container_width=True):
            auth.sign_out(account)
            cfg.google.account = ""
            save()
            st.rerun()
        st.caption("Use a personal Google account — work/school accounts aren't supported by "
                   "Gemini Notebook's API yet.")

    with st.expander("Zotero", expanded=not cfg.is_zotero_readable()):
        cfg.zotero.use_local_api = st.checkbox(
            "Read from Zotero desktop when it's running", value=cfg.zotero.use_local_api,
            help="Zotero 7: Settings → Advanced → allow other applications on this computer to "
                 "communicate with Zotero. No API key needed to send to Gemini Notebook.")
        cfg.zotero.api_key = st.text_input("API key (needed to bring things back into Zotero)",
                                           cfg.zotero.api_key, type="password")
        cfg.zotero.library_id = st.text_input("Library ID (your numeric user ID)", cfg.zotero.library_id)
        cfg.zotero.library_type = st.selectbox("Library type", ["user", "group"],
                                               index=0 if cfg.zotero.library_type == "user" else 1)
        cfg.zotero.local_storage_path = st.text_input(
            "Zotero storage folder (optional fallback)",
            cfg.zotero.local_storage_path or detect_storage_path())
        if st.button("Save & test Zotero"):
            save()
            ok, msg = ZoteroLibrary(cfg.zotero).test_connection()
            (st.success if ok else st.error)(msg)

    with st.expander("Sync settings"):
        tiers = list(TIER_LIMITS)
        cfg.sync.plan_tier = st.selectbox(
            "Your Gemini Notebook plan", tiers, index=tiers.index(cfg.sync.plan_tier),
            format_func=lambda t: f"{t.title()} — up to {TIER_LIMITS[t][1]} sources per notebook",
            help="Big collections are split across several notebooks to fit this limit.")
        cfg.sync.include_bibliography = st.checkbox(
            "Add a Bibliography source to each notebook", cfg.sync.include_bibliography,
            help="Lists every item with author-year, DOI and Zotero key, so Gemini cites accurately.")
        cfg.sync.include_annotations = st.checkbox(
            "Add my highlights & notes", cfg.sync.include_annotations)
        cfg.sync.cite_items_without_pdf = st.checkbox(
            "Include items without a PDF (web page or citation card)", cfg.sync.cite_items_without_pdf)
        if st.button("Save settings"):
            save()
            st.success("Saved")

    with st.expander("🧪 Experimental: read notes & text back"):
        st.caption("Gemini Notebook's official API can't read notes or source text yet. This "
                   "optional add-on uses an unofficial library and its own browser login, which "
                   "expires every few weeks.")
        cfg.experimental.enabled = st.checkbox("Enable", cfg.experimental.enabled)
        if cfg.experimental.enabled:
            ok, msg = ExperimentalReader.available(cfg.experimental.storage_path)
            (st.success if ok else st.warning)(msg)
            if ok and st.button("Check experimental login"):
                good, m = ExperimentalReader(cfg.experimental.storage_path).check()
                (st.success if good else st.error)(m)
        if st.button("Save", key="save_exp"):
            save()


# ══════════════════ MAIN ══════════════════

st.markdown('<p class="cb-title">Cite<span>Bridge</span></p>', unsafe_allow_html=True)
st.markdown('<p class="cb-sub">Zotero ↔ Gemini Notebook (formerly NotebookLM)</p>', unsafe_allow_html=True)

stats = state.stats() if account else {}
m = st.columns(5)
m[0].metric("Google", account.split("@")[0] if account else "Not signed in")
m[1].metric("Notebooks", stats.get("notebooks", 0))
m[2].metric("Sources", stats.get("sources", 0))
m[3].metric("Processing / failed", f"{stats.get('processing', 0)} / {stats.get('failed', 0)}")
m[4].metric("To clean up", stats.get("orphans", 0))

if not account:
    st.info("👈 Sign in with Google to get started.")
    st.stop()
if not cfg.is_zotero_readable():
    st.info("👈 Connect Zotero: open Zotero desktop, or add your API key.")
    st.stop()

if not ss.get("adopted"):
    try:
        adopted = engine().adopt_legacy()
        if adopted:
            st.success(f"Linked your existing notebooks from CiteBridge v1: {', '.join(adopted)}")
    except Exception as e:
        logging.warning(f"Legacy adoption skipped: {e}")
    ss.adopted = True

tab_send, tab_proc, tab_clean, tab_back, tab_dl, tab_hist = st.tabs([
    "📤 Send to Gemini Notebook", "⏳ Processing", "🧹 Cleanup",
    "📥 Bring back to Zotero", "💾 Download originals", "📋 History"])

# ── Send ──
with tab_send:
    try:
        collections = sorted(zot.collections(), key=lambda c: c.name.lower())
    except Exception as e:
        st.error(friendly(e))
        collections = []
    if collections:
        st.caption(f"Reading from {zot.read_mode_label}.")
        chosen = set(cfg.sync.enabled_collections)
        picked = []
        cols = st.columns(2)
        for i, c in enumerate(collections):
            if cols[i % 2].checkbox(f"**{c.name}** ({c.num_items} items)", c.key in chosen, key=f"c_{c.key}"):
                picked.append(c)

        if st.button("Preview what will be sent", disabled=not picked):
            cfg.sync.enabled_collections = [c.key for c in picked]
            save()
            ss.plans = []
            ss.plans_account = account
            with st.status("Planning…", expanded=True) as status:
                eng = engine(lambda msg: status.write(msg))
                for c in picked:
                    try:
                        ss.plans.append(eng.plan(c.key, c.name))
                    except Exception as e:
                        st.error(f"{c.name}: {friendly(e)}")
                status.update(label="Plan ready", state="complete")

        if ss.get("plans_account") != account:   # plans belong to the account that made them
            ss.plans = []
        plans = ss.get("plans") or []
        for p in plans:
            with st.container(border=True):
                st.markdown(f"**{p.collection_name}** — {p.summary()}")
                if p.notebooks:
                    st.caption("Notebooks: " + ", ".join(
                        t.display_name + (" (new)" if t.existing is None else "")
                        + (" (renamed)" if t.rename else "") for t in p.notebooks))
                for w in p.warnings:
                    st.warning(w)
                if p.actions or p.skips:
                    with st.expander("Details"):
                        for a in p.actions:
                            st.write(f"`{a.source_kind}` {a.display_name} — *{a.reason}*")
                        for s in p.skips:
                            st.write(f"⏭️ {s.title} — {s.reason}")
        if plans and any(not p.is_noop for p in plans):
            if st.button("📤 Send to Gemini Notebook", type="primary"):
                ss.plans = []   # consume first: an interrupted run must not be re-sent stale
                with st.status("Sending…", expanded=True) as status:
                    eng = engine(lambda msg: status.write(msg))
                    for p in plans:
                        try:
                            rep = eng.execute(p)
                            (st.success if rep.ok else st.warning)(rep.summary())
                            for f in rep.failed + rep.errors:
                                st.error(f)
                            for url in rep.notebooks:
                                st.markdown(f"[Open in Gemini Notebook ↗]({url})")
                        except Exception as e:
                            st.error(f"{p.collection_name}: {friendly(e)}")
                    status.update(label="Done", state="complete")
        elif plans:
            st.success("Everything selected is already up to date.")
    else:
        st.info("No Zotero collections found.")

# ── Processing ──
with tab_proc:
    if st.button("Check status now"):
        try:
            n = engine().refresh_pending()
            st.success(f"Updated {n} source(s).")
        except Exception as e:
            st.error(friendly(e))
    from citebridge.notebook.base import explain_failure
    pending, failed = state.pending_sources(), state.failed_sources()
    st.markdown(f"**Still processing:** {len(pending)}")
    for r in pending:
        st.write(f"⏳ {r.display_name}")
    st.markdown(f"**Failed:** {len(failed)}")
    for r in failed:
        st.write(f"❌ {r.display_name} — {explain_failure(r.error_code, r.error_message or '')}")

# ── Cleanup ──
with tab_clean:
    st.caption("Gemini Notebook's API can't delete sources yet. These outdated sources still take "
               "up space in your notebooks — delete them in Gemini Notebook, then mark them done "
               "(CiteBridge also notices automatically on your next sync).")
    orphans = state.orphans()
    if not orphans:
        st.success("Nothing to clean up.")
    for o in orphans:
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{o.display_name}** — {o.reason} · [open notebook ↗]({o.notebook_url})")
        if c2.button("Done", key=f"o_{o.source_name}"):
            state.resolve_orphan(o.source_name)
            st.rerun()

# ── Bring back ──
with tab_back:
    synced = state.all_notebooks()
    if not synced:
        st.info("Send a collection first.")
    elif not cfg.is_zotero_writable():
        st.warning("Add your Zotero API key and library ID (sidebar → Zotero) to write into Zotero.")
    else:
        names = {c.key: c.name for c in collections}
        rd = reader()
        for nb in synced:
            label = f"{names.get(nb.collection_key, nb.collection_key)} → {nb.display_name}"
            with st.container(border=True):
                st.markdown(f"**{label}**")
                c1, c2 = st.columns(2)
                if c1.button("Import sources I added in Gemini Notebook", key=f"i_{nb.notebook_name}"):
                    with st.spinner("Importing…"):
                        try:
                            rep = reverse.import_new_sources(nb.notebook_name, nb.display_name,
                                                             writer(), zot, state, rd)
                            st.success(rep.summary())
                        except Exception as e:
                            st.error(friendly(e))
                if rd is None:
                    c2.caption("Pulling notes needs the 🧪 experimental add-on.")
                elif c2.button("🧪 Pull notes into Zotero", key=f"n_{nb.notebook_name}"):
                    with st.spinner("Pulling notes…"):
                        try:
                            rep = reverse.pull_notes(nb.collection_key, names.get(nb.collection_key, ""),
                                                     rd, zot, state)
                            st.success(rep.summary())
                        except Exception as e:
                            st.error(friendly(e))

# ── Download ──
with tab_dl:
    st.caption("Saves the original web pages/files behind a notebook's web sources.")
    if st.button("Load my notebooks"):
        try:
            ss.notebook_list = writer().list_notebooks()
        except Exception as e:
            st.error(friendly(e))
    nbs = ss.get("notebook_list") or []
    if nbs:
        nb = st.selectbox("Notebook", nbs, format_func=lambda n: n.display_name)
        out = st.text_input("Save to", str(Path.home() / "CiteBridge Downloads"))
        if st.button("Download original files"):
            with st.spinner("Downloading…"):
                rep = reverse.download_originals(nb.name, nb.display_name, writer(), out)
            st.success(f"{rep.imported} saved, {rep.skipped} skipped (uploaded files or YouTube).")
            for e in rep.errors:
                st.error(e)

# ── History ──
with tab_hist:
    for ts, action, status, details in state.recent_logs(30):
        icon = {"success": "✅", "partial": "⚠️"}.get(status, "❌")
        st.markdown(f"{icon} **{action}** · {ts}  \n{details or ''}")

st.divider()
st.caption(f"CiteBridge {__version__} · Built by Rob Voss, Ph.D. · "
           "[Voss AI Consulting](https://www.vossaiconsulting.com) · "
           "Independent tool, not affiliated with Zotero or Google.")
