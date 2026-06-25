"""
CiteBridge — Streamlit GUI
One-button bidirectional sync between Zotero and NotebookLM.

Run with:  streamlit run app.py
"""

import streamlit as st
import logging
import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from config import AppConfig
from sync_engine import SyncEngine
from state_db import SyncStateDB
from notebooklm_client import NotebookLMClient
from zotero_client import ZoteroClient
from utils import setup_logging, get_zotero_storage_path

setup_logging()

# ── Page Config ──
st.set_page_config(
    page_title="CiteBridge",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #888;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .status-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px 20px;
        margin: 5px 0;
        border-left: 4px solid #667eea;
    }
    .sync-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ──
if "config" not in st.session_state:
    st.session_state.config = AppConfig.load()
if "sync_log" not in st.session_state:
    st.session_state.sync_log = []
if "is_syncing" not in st.session_state:
    st.session_state.is_syncing = False
if "zotero_connected" not in st.session_state:
    st.session_state.zotero_connected = False
if "nlm_connected" not in st.session_state:
    st.session_state.nlm_connected = False


def add_log(msg: str):
    """Add a message to the sync log."""
    st.session_state.sync_log.append(msg)


# ══════════════════════════════════════════
# SIDEBAR — Settings & Authentication
# ══════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # ── Zotero Settings ──
    with st.expander("🔶 Zotero Connection", expanded=not st.session_state.config.is_zotero_configured()):
        st.markdown(
            "Get your API key and Library ID from "
            "[zotero.org/settings/keys](https://www.zotero.org/settings/keys)"
        )
        api_key = st.text_input(
            "API Key",
            value=st.session_state.config.zotero.api_key,
            type="password",
            key="zotero_api_key",
        )
        library_id = st.text_input(
            "Library ID (your user ID number)",
            value=st.session_state.config.zotero.library_id,
            key="zotero_library_id",
        )
        library_type = st.selectbox(
            "Library Type",
            ["user", "group"],
            index=0 if st.session_state.config.zotero.library_type == "user" else 1,
            key="zotero_library_type",
        )

        # Auto-detect Zotero storage
        auto_storage = get_zotero_storage_path()
        storage_path = st.text_input(
            "Local Storage Path (optional, for faster PDF access)",
            value=st.session_state.config.zotero.local_storage_path or auto_storage,
            key="zotero_storage_path",
            help="Path to your Zotero/storage/ directory",
        )

        if st.button("💾 Save & Test Zotero", key="save_zotero"):
            st.session_state.config.zotero.api_key = api_key
            st.session_state.config.zotero.library_id = library_id
            st.session_state.config.zotero.library_type = library_type
            st.session_state.config.zotero.local_storage_path = storage_path
            st.session_state.config.save()

            # Test connection
            try:
                zot = ZoteroClient(library_id, api_key, library_type, storage_path)
                if zot.test_connection():
                    st.success("✅ Zotero connected!")
                    st.session_state.zotero_connected = True
                else:
                    st.error("❌ Connection failed — check your API key and Library ID")
                    st.session_state.zotero_connected = False
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.session_state.zotero_connected = False

    # ── NotebookLM Settings ──
    with st.expander("🟣 NotebookLM Connection", expanded=not NotebookLMClient.is_authenticated()):
        st.markdown(
            "NotebookLM authentication uses your Google account cookies.\n\n"
            "**Option A — Local usage:** Run `notebooklm login` in your terminal.\n\n"
            "**Option B — Streamlit Cloud:** Add `NOTEBOOKLM_AUTH_JSON` to your "
            "[app secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) "
            "with the contents of `~/.notebooklm/storage_state.json`."
        )

        # Auth diagnostics
        _has_env = bool(os.environ.get("NOTEBOOKLM_AUTH_JSON", "").strip())
        _has_st_secret = False
        try:
            _secret = st.secrets.get("NOTEBOOKLM_AUTH_JSON", "")
            _has_st_secret = bool(_secret and str(_secret).strip())
        except Exception:
            pass
        _has_file = False
        try:
            from notebooklm.paths import get_storage_path as _gsp
            _has_file = _gsp().exists()
        except Exception:
            pass

        st.caption(
            f"Auth sources: env var {'✅' if _has_env else '❌'} · "
            f"Streamlit secret {'✅' if _has_st_secret else '❌'} · "
            f"local file {'✅' if _has_file else '❌'}"
        )

        nlm_storage = st.text_input(
            "Auth Storage Path (leave blank for default)",
            value=st.session_state.config.notebooklm.storage_path,
            key="nlm_storage_path",
            help="Default: ~/.notebooklm/storage_state.json",
        )

        if st.button("🔗 Verify NotebookLM Connection", key="verify_nlm"):
            st.session_state.config.notebooklm.storage_path = nlm_storage
            st.session_state.config.save()

            try:
                nlm = NotebookLMClient(nlm_storage or None)
                if nlm.test_connection():
                    st.success("✅ NotebookLM connected!")
                    st.session_state.nlm_connected = True
                else:
                    st.error("❌ Not authenticated — check auth sources above")
                    st.session_state.nlm_connected = False
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.session_state.nlm_connected = False

    # ── Sync Settings ──
    with st.expander("🔄 Sync Options"):
        sync_notes_back = st.checkbox(
            "Sync NotebookLM notes back to Zotero",
            value=st.session_state.config.sync.sync_notes_back,
            key="sync_notes_back",
        )
        max_file_size = st.slider(
            "Max file size (MB)",
            min_value=10, max_value=500, step=10,
            value=st.session_state.config.sync.max_file_size_mb,
            key="max_file_size",
        )

        if st.button("💾 Save Sync Settings", key="save_sync"):
            st.session_state.config.sync.sync_notes_back = sync_notes_back
            st.session_state.config.sync.max_file_size_mb = max_file_size
            st.session_state.config.save()
            st.success("Settings saved!")


# ══════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════

st.markdown('<p class="main-header">🔗 CiteBridge</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Bidirectional Zotero ↔ NotebookLM Sync</p>',
    unsafe_allow_html=True,
)

# ── Connection Status ──
col1, col2, col3 = st.columns(3)
with col1:
    zotero_ok = st.session_state.config.is_zotero_configured()
    st.metric("Zotero", "✅ Connected" if zotero_ok else "❌ Not configured")
with col2:
    # "Authed" = credentials are present (file / env var / Streamlit secret).
    # "Verified" = a real API call succeeded this session. File presence alone
    # does NOT mean the connection works — Google tokens expire periodically,
    # so we never report "Connected" until a live check has passed.
    nlm_authed = NotebookLMClient.is_authenticated()
    nlm_verified = st.session_state.get("nlm_connected", False)
    if nlm_verified:
        nlm_status = "✅ Connected"
    elif nlm_authed:
        nlm_status = "🔑 Verify needed"
    else:
        nlm_status = "❌ Not authenticated"
    st.metric("NotebookLM", nlm_status)
with col3:
    db = SyncStateDB()
    stats = db.get_sync_stats()
    st.metric("Items Synced", stats["items_synced"])

st.divider()

# ── Main Tabs ──
if not zotero_ok:
    st.warning(
        "👈 Configure your Zotero API key in the sidebar to get started."
    )
elif not nlm_authed:
    st.warning(
        "👈 Authenticate with NotebookLM — run `notebooklm login` in your "
        "terminal, then click **Verify NotebookLM Connection** in the sidebar."
    )
else:
    if not nlm_verified:
        st.info(
            "🔑 NotebookLM credentials found but **not verified** this session. "
            "Click **Verify NotebookLM Connection** in the sidebar before "
            "syncing. NotebookLM tokens expire periodically — if verification "
            "fails, re-run `notebooklm login` in your terminal."
        )
    tab_push, tab_import, tab_download, tab_history = st.tabs([
        "📤 Push to NotebookLM",
        "📥 Import Sources to Zotero",
        "💾 Download Source Files",
        "📋 Sync History",
    ])

    # ══════════════════════════════════════
    # TAB 1: Zotero → NotebookLM
    # ══════════════════════════════════════
    with tab_push:
        try:
            zot = ZoteroClient(
                st.session_state.config.zotero.library_id,
                st.session_state.config.zotero.api_key,
                st.session_state.config.zotero.library_type,
                st.session_state.config.zotero.local_storage_path,
            )
            collections = zot.get_collections()

            if not collections:
                st.info("No collections found in your Zotero library.")
            else:
                st.markdown("### 📚 Select Zotero Collections to Push")
                st.markdown(
                    "Each selected collection becomes a NotebookLM notebook "
                    "with all PDFs added as sources."
                )

                previously_selected = set(
                    st.session_state.config.sync.enabled_collections
                )

                selected_keys = []
                cols = st.columns(2)
                for i, coll in enumerate(sorted(collections, key=lambda c: c.name)):
                    with cols[i % 2]:
                        checked = st.checkbox(
                            f"📁 **{coll.name}** ({coll.num_items} items)",
                            value=coll.key in previously_selected,
                            key=f"coll_{coll.key}",
                        )
                        if checked:
                            selected_keys.append(coll.key)

                st.divider()

                col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                with col_btn2:
                    sync_clicked = st.button(
                        "📤 **PUSH TO NOTEBOOKLM**",
                        type="primary",
                        use_container_width=True,
                        disabled=len(selected_keys) == 0,
                        key="sync_now",
                    )

                if len(selected_keys) == 0:
                    st.caption("Select at least one collection to push.")

                if sync_clicked and selected_keys:
                    st.session_state.config.sync.enabled_collections = selected_keys
                    st.session_state.config.save()
                    st.session_state.sync_log = []

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    step_count = [0]
                    total_steps = len(selected_keys) * 3

                    def progress_callback_push(msg):
                        step_count[0] += 1
                        progress = min(step_count[0] / max(total_steps, 1), 0.99)
                        progress_bar.progress(progress)
                        status_text.markdown(f"**{msg}**")
                        st.session_state.sync_log.append(msg)

                    try:
                        engine = SyncEngine(
                            st.session_state.config,
                            progress_callback=progress_callback_push,
                        )
                        result = engine.sync_all(selected_keys)
                        progress_bar.progress(1.0)
                        status_text.empty()

                        if result.success:
                            st.success(f"✅ Push complete!\n\n{result.summary()}")
                        else:
                            st.warning(
                                f"⚠️ Push completed with errors:\n\n{result.summary()}"
                            )
                            for err in result.errors:
                                st.error(err)
                    except Exception as e:
                        st.error(f"❌ Push failed: {e}")
                        logging.exception("Push failed")

        except Exception as e:
            st.error(f"❌ Failed to load Zotero collections: {e}")
            st.info("Check your Zotero API key and Library ID in the sidebar.")

    # ══════════════════════════════════════
    # TAB 2: NotebookLM → Zotero (SOURCES)
    # ══════════════════════════════════════
    with tab_import:
        st.markdown("### 📥 Import NotebookLM Sources into Zotero")
        st.markdown(
            "Select notebooks below to pull their **raw sources** "
            "(PDFs, web pages, videos, etc.) into your Zotero library. "
            "Each notebook becomes a Zotero collection with proper "
            "library items, URLs, and full extracted text."
        )

        try:
            nlm = NotebookLMClient(
                st.session_state.config.notebooklm.storage_path or None,
            )
            notebooks = nlm.list_notebooks(include_source_counts=True)

            if not notebooks:
                st.info("No notebooks found in NotebookLM.")
            else:
                selected_notebooks = []
                cols = st.columns(2)

                for i, nb in enumerate(sorted(notebooks, key=lambda n: n.title)):
                    with cols[i % 2]:
                        checked = st.checkbox(
                            f"📓 **{nb.title}** ({nb.sources_count} sources)",
                            key=f"nb_{nb.id}",
                        )
                        if checked:
                            selected_notebooks.append(nb.id)

                st.divider()

                include_fulltext = st.checkbox(
                    "Include full extracted text (slower but much more valuable)",
                    value=True,
                    key="include_fulltext",
                    help="Fetches the complete text NotebookLM extracted from "
                         "each source and saves it as a Zotero note. "
                         "This is the main value — you get the full content "
                         "even if the original source goes offline.",
                )

                col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                with col_btn2:
                    import_clicked = st.button(
                        "📥 **IMPORT SOURCES TO ZOTERO**",
                        type="primary",
                        use_container_width=True,
                        disabled=len(selected_notebooks) == 0,
                        key="import_now",
                    )

                if len(selected_notebooks) == 0:
                    st.caption("Select at least one notebook to import.")

                if import_clicked and selected_notebooks:
                    st.session_state.sync_log = []

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    step_count = [0]
                    total_steps = len(selected_notebooks) * 5

                    def progress_callback_import(msg):
                        step_count[0] += 1
                        progress = min(step_count[0] / max(total_steps, 1), 0.99)
                        progress_bar.progress(progress)
                        status_text.markdown(f"**{msg}**")
                        st.session_state.sync_log.append(msg)

                    try:
                        engine = SyncEngine(
                            st.session_state.config,
                            progress_callback=progress_callback_import,
                        )
                        result = engine.import_all_notebooks(
                            selected_notebooks,
                            include_fulltext=include_fulltext,
                        )
                        progress_bar.progress(1.0)
                        status_text.empty()

                        if result.success:
                            st.success(
                                f"✅ Import complete!\n\n{result.summary()}\n\n"
                                f"Open Zotero to see your new collections "
                                f"(prefixed with 'NLM:')."
                            )
                        else:
                            st.warning(
                                f"⚠️ Import completed with errors:\n\n"
                                f"{result.summary()}"
                            )
                            for err in result.errors:
                                st.error(err)
                    except Exception as e:
                        st.error(f"❌ Import failed: {e}")
                        logging.exception("Import failed")

        except Exception as e:
            st.error(f"❌ Failed to load NotebookLM notebooks: {e}")
            st.info(
                "Make sure you've authenticated with NotebookLM. "
                "Run `notebooklm login` in your terminal."
            )

    # ══════════════════════════════════════
    # TAB 3: Download Source Files
    # ══════════════════════════════════════
    with tab_download:
        st.markdown("### 💾 Download Source Files from NotebookLM")
        st.markdown(
            "Download the **original files** (PDFs, web pages, etc.) from "
            "your NotebookLM notebooks to a local directory. "
            "This is especially useful for sources added by NotebookLM's "
            "research features that you don't have locally."
        )

        try:
            nlm_dl = NotebookLMClient(
                st.session_state.config.notebooklm.storage_path or None,
            )
            dl_notebooks = nlm_dl.list_notebooks(include_source_counts=True)

            if not dl_notebooks:
                st.info("No notebooks found in NotebookLM.")
            else:
                selected_dl_notebooks = []
                dl_cols = st.columns(2)

                for i, nb in enumerate(sorted(dl_notebooks, key=lambda n: n.title)):
                    with dl_cols[i % 2]:
                        checked = st.checkbox(
                            f"📓 **{nb.title}** ({nb.sources_count} sources)",
                            key=f"dl_{nb.id}",
                        )
                        if checked:
                            selected_dl_notebooks.append(nb.id)

                st.divider()

                # Output directory selection
                default_dir = str(Path.home() / "CiteBridge Downloads")
                output_dir = st.text_input(
                    "Download directory",
                    value=default_dir,
                    key="download_dir",
                    help="Each notebook gets its own subdirectory inside this folder.",
                )

                dl_timeout = st.slider(
                    "Download timeout per file (seconds)",
                    min_value=10, max_value=120, value=30, step=5,
                    key="dl_timeout",
                    help="Increase this if downloads are timing out for large files.",
                )

                col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
                with col_dl2:
                    download_clicked = st.button(
                        "💾 **DOWNLOAD SOURCE FILES**",
                        type="primary",
                        use_container_width=True,
                        disabled=len(selected_dl_notebooks) == 0,
                        key="download_now",
                    )

                if len(selected_dl_notebooks) == 0:
                    st.caption("Select at least one notebook to download from.")

                if download_clicked and selected_dl_notebooks:
                    st.session_state.sync_log = []

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    step_count = [0]
                    total_steps = len(selected_dl_notebooks) * 5

                    def progress_callback_download(msg):
                        step_count[0] += 1
                        progress = min(step_count[0] / max(total_steps, 1), 0.99)
                        progress_bar.progress(progress)
                        status_text.markdown(f"**{msg}**")
                        st.session_state.sync_log.append(msg)

                    try:
                        engine = SyncEngine(
                            st.session_state.config,
                            progress_callback=progress_callback_download,
                        )
                        dl_result = engine.download_all_notebooks(
                            selected_dl_notebooks,
                            output_dir=output_dir,
                            timeout=float(dl_timeout),
                        )
                        progress_bar.progress(1.0)
                        status_text.empty()

                        # Show results
                        downloaded = dl_result.items_uploaded
                        skipped = dl_result.items_skipped
                        errors = len(dl_result.errors)

                        if dl_result.success:
                            st.success(
                                f"✅ Download complete!\n\n"
                                f"**{downloaded}** files downloaded, "
                                f"**{skipped}** skipped (no URL)\n\n"
                                f"Saved to: `{output_dir}`"
                            )
                        else:
                            st.warning(
                                f"⚠️ Download completed with issues:\n\n"
                                f"**{downloaded}** downloaded, "
                                f"**{skipped}** skipped, "
                                f"**{errors}** failed"
                            )
                            for err in dl_result.errors:
                                st.error(err)

                        # Show per-notebook breakdown
                        if dl_result.log:
                            with st.expander("📋 Per-Notebook Details"):
                                for entry in dl_result.log:
                                    st.markdown(entry)

                    except Exception as e:
                        st.error(f"❌ Download failed: {e}")
                        logging.exception("Download failed")

        except Exception as e:
            st.error(f"❌ Failed to load NotebookLM notebooks: {e}")
            st.info(
                "Make sure you've authenticated with NotebookLM. "
                "Run `notebooklm login` in your terminal."
            )

    # ══════════════════════════════════════
    # TAB 4: History
    # ══════════════════════════════════════
    with tab_history:
        st.markdown("### 📋 Sync History")

        db = SyncStateDB()
        logs = db.get_recent_logs(20)

        if logs:
            for entry in logs:
                icon = (
                    "✅" if entry.status == "success"
                    else "⚠️" if entry.status == "partial"
                    else "❌"
                )
                st.markdown(
                    f"**{icon} {entry.action}** — {entry.timestamp}\n\n"
                    f"<small>{entry.details}</small>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No sync history yet. Run your first sync above!")


# ── Live Log (from current session) ──
if st.session_state.sync_log:
    with st.expander("📜 Detailed Log (Current Session)", expanded=False):
        for msg in st.session_state.sync_log:
            st.text(msg)


# ── Footer ──
st.divider()
st.markdown(
    "<center>"
    "<small>CiteBridge v1.0 — Built by Rob Voss, Ph.D. / "
    "<a href='https://www.vossaiconsulting.com'>Voss AI Consulting</a>"
    " &bull; <a href='https://www.robvoss.com'>robvoss.com</a></small><br>"
    "<small>Custom AI integrations for business, education, and finance</small>"
    "</center>",
    unsafe_allow_html=True,
)
