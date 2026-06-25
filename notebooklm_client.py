"""
NotebookLM API client — wraps notebooklm-py for creating notebooks,
adding sources, and reading notes.

Uses the unofficial notebooklm-py library (async API).
All public methods in this wrapper are synchronous for easy integration.

Key design: Each API call creates a fresh async client within a proper
async context manager, ensuring the httpx session is always opened and
closed within the same event loop.  Auth tokens are fetched once and
cached for reuse across calls.
"""

import asyncio
import logging
import mimetypes
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Data Classes (stable interface for rest of app) ──

@dataclass
class NLMNotebook:
    """Represents a NotebookLM notebook."""
    id: str
    title: str
    sources_count: int = 0
    created_at: str = ""


@dataclass
class NLMSource:
    """Represents a source in a NotebookLM notebook."""
    id: str
    title: str
    source_type: str = ""
    status: str = ""
    is_ready: bool = False
    url: Optional[str] = None


@dataclass
class NLMSourceFull:
    """A source with its full extracted text content."""
    id: str
    title: str
    source_type: str = ""
    url: Optional[str] = None
    content: str = ""
    char_count: int = 0


@dataclass
class NLMSourceDownloadResult:
    """Result of downloading a single source file."""
    source_id: str
    title: str
    url: str = ""
    file_path: str = ""
    success: bool = False
    error: str = ""
    content_type: str = ""
    size_bytes: int = 0


@dataclass
class NLMNote:
    """Represents a note in a NotebookLM notebook."""
    id: str
    title: str
    content: str = ""


# ── Async-to-Sync Bridge ──

def _run_async(coro):
    """Run an async coroutine synchronously.

    Handles the case where we're inside an existing event loop
    (e.g., Streamlit) by running in a separate thread with its
    own event loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an existing event loop (e.g., Streamlit).
        # Run in a thread with a fresh event loop.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


# ── Main Client ──

class NotebookLMClient:
    """
    High-level synchronous client for NotebookLM.
    Wraps the async notebooklm-py library (v0.3.x).

    Each API call creates a fresh async client connection within a
    proper async context manager.  This avoids event-loop and session
    lifecycle issues that arise when mixing sync/async code.

    Auth tokens (cookies + CSRF + session ID) are fetched once from
    storage and reused across calls.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Args:
            storage_path: Optional path to storage_state.json.
                          If None, uses default ~/.notebooklm/ location.
                          Also supports NOTEBOOKLM_AUTH_JSON env var.
        """
        self._storage_path = storage_path
        self._auth = None   # Cached AuthTokens (fetched once)

    # ── Internal Helpers ──

    @staticmethod
    def _load_streamlit_secret():
        """Try to load auth JSON from Streamlit secrets into env var.

        Streamlit Cloud stores secrets in st.secrets, which doesn't
        always automatically appear in os.environ.  This bridge
        ensures notebooklm-py's AuthTokens.from_storage() can find it.
        """
        if os.environ.get("NOTEBOOKLM_AUTH_JSON", "").strip():
            return  # Already set

        try:
            import streamlit as st
            auth_json = st.secrets.get("NOTEBOOKLM_AUTH_JSON", "")
            if auth_json and str(auth_json).strip():
                os.environ["NOTEBOOKLM_AUTH_JSON"] = str(auth_json).strip()
                logger.info("Loaded NotebookLM auth from Streamlit secrets")
        except Exception:
            pass  # Not running in Streamlit or no secret set

    def _ensure_auth(self):
        """Fetch auth tokens from storage (done once, then cached)."""
        if self._auth is not None:
            return

        # Bridge Streamlit secrets → env var before auth lookup
        self._load_streamlit_secret()

        async def _fetch_auth():
            from notebooklm.auth import AuthTokens
            path = Path(self._storage_path) if self._storage_path else None
            return await AuthTokens.from_storage(path)

        try:
            self._auth = _run_async(_fetch_auth())
            logger.info("NotebookLM authentication tokens loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NotebookLM auth: {e}")
            raise

    def _call(self, async_fn):
        """Execute an async operation with a properly managed client.

        Creates a fresh NotebookLMClient (from notebooklm-py) for each
        call, opening and closing the httpx session within the same
        event loop.

        Args:
            async_fn: An async callable that takes the opened client
                      and returns a result.
        """
        self._ensure_auth()

        async def _execute():
            from notebooklm.client import NotebookLMClient as _AsyncClient
            async with _AsyncClient(self._auth) as client:
                result = await async_fn(client)
                # Capture any refreshed tokens
                self._auth = client.auth
                return result

        return _run_async(_execute())

    # ── Connection & Auth ──

    def test_connection(self) -> bool:
        """Test if we can connect to NotebookLM."""
        try:
            notebooks = self.list_notebooks()
            logger.info(
                f"NotebookLM connection OK — {len(notebooks)} notebooks found"
            )
            return True
        except Exception as e:
            logger.error(f"NotebookLM connection test failed: {e}")
            return False

    @staticmethod
    def login():
        """
        Launch the interactive login flow.
        This opens a browser for Google OAuth authentication.
        """
        try:
            cli_path = "notebooklm"
            for p in [
                Path.home() / ".local" / "bin" / "notebooklm",
                Path(sys.prefix) / "bin" / "notebooklm",
            ]:
                if p.exists():
                    cli_path = str(p)
                    break

            logger.info("Launching NotebookLM login flow...")
            result = subprocess.run(
                [cli_path, "login"],
                capture_output=False,
                timeout=300,
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False

    @staticmethod
    def is_authenticated() -> bool:
        """Check if NotebookLM auth tokens exist (file, env var, or Streamlit secret)."""
        try:
            # Check env var (for CI / direct deployment)
            auth_json = os.environ.get("NOTEBOOKLM_AUTH_JSON", "").strip()
            if auth_json:
                return True

            # Check Streamlit secrets (for Streamlit Cloud)
            try:
                import streamlit as st
                secret = st.secrets.get("NOTEBOOKLM_AUTH_JSON", "")
                if secret and str(secret).strip():
                    return True
            except Exception:
                pass

            # Check file-based auth (local usage)
            from notebooklm.paths import get_storage_path
            return get_storage_path().exists()
        except Exception:
            return False

    # ══════════════════════════════════════════
    #  Notebook Operations
    # ══════════════════════════════════════════

    def list_notebooks(self, include_source_counts: bool = False) -> List[NLMNotebook]:
        """List all notebooks in the account.

        Args:
            include_source_counts: If True, fetches the actual source count
                for each notebook (slower but accurate).  The NotebookLM API
                does not include source counts in notebook metadata, so this
                requires one extra API call per notebook.
        """
        if include_source_counts:
            # Single client session for all calls (much faster)
            async def _op_with_counts(client):
                notebooks = await client.notebooks.list()
                results = []
                for nb in notebooks:
                    try:
                        sources = await client.sources.list(nb.id)
                        count = len(sources)
                    except Exception:
                        count = 0
                    results.append((nb, count))
                return results

            raw = self._call(_op_with_counts)
            return [
                NLMNotebook(
                    id=nb.id,
                    title=nb.title,
                    sources_count=count,
                    created_at=str(nb.created_at) if nb.created_at else "",
                )
                for nb, count in raw
            ]
        else:
            async def _op(client):
                return await client.notebooks.list()

            raw = self._call(_op)
            return [
                NLMNotebook(
                    id=nb.id,
                    title=nb.title,
                    sources_count=0,
                    created_at=str(nb.created_at) if nb.created_at else "",
                )
                for nb in raw
            ]

    def create_notebook(self, title: str) -> NLMNotebook:
        """Create a new notebook."""
        async def _op(client):
            return await client.notebooks.create(title)

        nb = self._call(_op)
        result = NLMNotebook(id=nb.id, title=nb.title)
        logger.info(f"Created notebook: {result.title} ({result.id})")
        return result

    def find_notebook_by_title(self, title: str) -> Optional[NLMNotebook]:
        """Find a notebook by exact title match."""
        notebooks = self.list_notebooks()
        for nb in notebooks:
            if nb.title == title:
                return nb
        return None

    def find_or_create_notebook(self, title: str) -> NLMNotebook:
        """Find existing notebook by title, or create a new one."""
        existing = self.find_notebook_by_title(title)
        if existing:
            logger.info(f"Found existing notebook: {title}")
            return existing
        return self.create_notebook(title)

    # ══════════════════════════════════════════
    #  Source Operations
    # ══════════════════════════════════════════

    def add_pdf_source(self, notebook_id: str, file_path: str,
                       wait: bool = True, timeout: float = 120.0) -> NLMSource:
        """
        Upload a PDF file as a source to a notebook.

        Args:
            notebook_id: The notebook to add the source to.
            file_path: Local path to the PDF file.
            wait: If True, wait for source to finish processing.
            timeout: Max seconds to wait for processing.
        """
        async def _op(client):
            return await client.sources.add_file(
                notebook_id, file_path,
                wait=wait, wait_timeout=timeout,
            )

        src = self._call(_op)
        result = NLMSource(
            id=src.id,
            title=src.title or Path(file_path).stem,
            source_type=str(src.kind),
            status=str(src.status),
            is_ready=src.is_ready,
        )
        logger.info(f"Added source: {result.title} → notebook {notebook_id}")
        return result

    def add_url_source(self, notebook_id: str, url: str,
                       wait: bool = True) -> NLMSource:
        """Add a URL as a source to a notebook."""
        async def _op(client):
            return await client.sources.add_url(
                notebook_id, url, wait=wait,
            )

        src = self._call(_op)
        return NLMSource(
            id=src.id,
            title=src.title or url,
            source_type=str(src.kind),
            is_ready=src.is_ready,
        )

    def list_sources(self, notebook_id: str) -> List[NLMSource]:
        """List all sources in a notebook."""
        async def _op(client):
            return await client.sources.list(notebook_id)

        raw = self._call(_op)
        return [
            NLMSource(
                id=s.id,
                title=s.title or "Untitled",
                source_type=str(s.kind),
                status=str(s.status),
                is_ready=s.is_ready,
                url=s.url,
            )
            for s in raw
        ]

    def get_source_content(self, notebook_id: str, source_id: str,
                           source: Optional[NLMSource] = None) -> NLMSourceFull:
        """
        Get the best available text content for a source.

        Primary path: the full extracted text via ``sources.get_fulltext()``
        — this is the complete text NotebookLM extracted from the source and
        is the real value of an import. If that is empty or unavailable, we
        fall back to the AI-generated summary + keywords from the source guide.

        Args:
            notebook_id: The notebook the source belongs to.
            source_id: The source to fetch content for.
            source: Optional pre-fetched NLMSource (title/type/url). Pass this
                    when iterating a known source list to avoid re-listing the
                    whole notebook for every source.
        """
        if source is None:
            for s in self.list_sources(notebook_id):
                if s.id == source_id:
                    source = s
                    break

        content = ""

        # Primary: real extracted full text
        try:
            full = self.get_source_fulltext(notebook_id, source_id)
            content = (getattr(full, "content", "") or "").strip()
        except Exception as e:
            logger.warning(f"get_fulltext failed for {source_id}: {e}")

        # Fallback: AI summary + keywords from the source guide
        if not content:
            try:
                guide = self.get_source_guide(notebook_id, source_id)
                summary, keywords = self._guide_summary_keywords(guide)
                parts = []
                if summary:
                    parts.append(summary)
                if keywords:
                    parts.append("\n\nKeywords: " + ", ".join(keywords))
                content = "".join(parts)
            except Exception as e:
                logger.warning(f"get_guide fallback failed for {source_id}: {e}")

        return NLMSourceFull(
            id=source_id,
            title=source.title if source else "",
            source_type=source.source_type if source else "",
            url=source.url if source else None,
            content=content,
            char_count=len(content),
        )

    def get_source_fulltext(self, notebook_id: str, source_id: str):
        """Get the full extracted text for a source (SourceFulltext object)."""
        async def _op(client):
            return await client.sources.get_fulltext(notebook_id, source_id)

        return self._call(_op)

    def get_source_guide(self, notebook_id: str,
                          source_id: str) -> Dict[str, Any]:
        """Get the AI-generated study guide for a source."""
        async def _op(client):
            return await client.sources.get_guide(notebook_id, source_id)

        return self._call(_op)

    @staticmethod
    def _guide_summary_keywords(guide: Any):
        """Read (summary, keywords) from a source guide.

        Tolerant of the two shapes notebooklm-py has returned: a plain dict
        (0.3.x–0.5.x) and a typed ``SourceGuide`` dataclass (0.6.0+).
        """
        if guide is None:
            return "", []
        if isinstance(guide, dict):
            return guide.get("summary", "") or "", list(guide.get("keywords", []) or [])
        summary = getattr(guide, "summary", "") or ""
        keywords = getattr(guide, "keywords", []) or []
        return summary, list(keywords)

    def get_all_sources_with_content(self, notebook_id: str
                                      ) -> List[NLMSourceFull]:
        """
        Get all sources in a notebook with their AI-generated summaries.
        This is the primary method for importing a NotebookLM research
        collection into Zotero.
        """
        sources = self.list_sources(notebook_id)
        results = []

        for src in sources:
            try:
                full = self.get_source_content(notebook_id, src.id, source=src)
                # Merge the URL from list_sources if content didn't have it
                if not full.url and src.url:
                    full.url = src.url
                if not full.source_type and src.source_type:
                    full.source_type = src.source_type
                results.append(full)
                logger.info(
                    f"Got content for: {full.title} ({full.char_count} chars)"
                )
            except Exception as e:
                logger.error(f"Failed to get content for {src.title}: {e}")
                # Still include with basic info
                results.append(NLMSourceFull(
                    id=src.id, title=src.title,
                    source_type=src.source_type, url=src.url,
                ))

        return results

    # ══════════════════════════════════════════
    #  Note Operations
    # ══════════════════════════════════════════

    def list_notes(self, notebook_id: str) -> List[NLMNote]:
        """List all notes in a notebook."""
        async def _op(client):
            return await client.notes.list(notebook_id)

        raw = self._call(_op)
        return [
            NLMNote(
                id=n.id,
                title=n.title,
                content=n.content,
            )
            for n in raw
        ]

    # ══════════════════════════════════════════
    #  Source Download Operations
    # ══════════════════════════════════════════

    @staticmethod
    def _sanitize_filename(name: str, max_len: int = 200) -> str:
        """Convert a source title to a safe filename."""
        # Remove/replace invalid characters
        safe = re.sub(r'[<>:"/\\|?*]', '_', name)
        safe = safe.strip('. ')
        if not safe:
            safe = "untitled"
        if len(safe) > max_len:
            safe = safe[:max_len].rstrip('. ')
        return safe

    @staticmethod
    def _guess_extension(content_type: str, url: str) -> str:
        """Guess a file extension from content type or URL."""
        # Try content type first
        if content_type:
            ext = mimetypes.guess_extension(content_type.split(';')[0].strip())
            if ext and ext != '.bin':
                return ext

        # Try URL
        path = url.split('?')[0].split('#')[0]
        if '.' in path.split('/')[-1]:
            ext = '.' + path.split('/')[-1].rsplit('.', 1)[-1].lower()
            if len(ext) <= 6:
                return ext

        # Defaults based on common content types
        ct = (content_type or '').lower()
        if 'pdf' in ct:
            return '.pdf'
        if 'html' in ct:
            return '.html'
        if 'json' in ct:
            return '.json'
        if 'xml' in ct:
            return '.xml'
        if 'text' in ct:
            return '.txt'

        return '.html'  # Default for web sources

    def download_source_file(self, source: NLMSource,
                              output_dir: str,
                              timeout: float = 30.0
                              ) -> NLMSourceDownloadResult:
        """
        Download a single source file from its original URL.

        Args:
            source: The NLMSource to download.
            output_dir: Directory to save the file.
            timeout: HTTP request timeout in seconds.

        Returns:
            NLMSourceDownloadResult with download details.
        """
        result = NLMSourceDownloadResult(
            source_id=source.id,
            title=source.title,
            url=source.url or "",
        )

        if not source.url:
            result.error = "No URL available (uploaded file — cannot download)"
            return result

        try:
            import httpx

            with httpx.Client(
                timeout=timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; CiteBridge/1.0)",
                },
            ) as http:
                response = http.get(source.url)
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                result.content_type = content_type

                # Determine filename
                ext = self._guess_extension(content_type, source.url)
                safe_name = self._sanitize_filename(source.title)
                if not safe_name.lower().endswith(ext.lower()):
                    safe_name += ext

                # Handle duplicate filenames
                out_path = Path(output_dir) / safe_name
                counter = 2
                base = out_path.stem
                while out_path.exists():
                    out_path = Path(output_dir) / f"{base} ({counter}){ext}"
                    counter += 1

                # Write file
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as f:
                    f.write(response.content)

                result.file_path = str(out_path)
                result.size_bytes = len(response.content)
                result.success = True
                logger.info(
                    f"Downloaded: {source.title} → {out_path.name} "
                    f"({result.size_bytes:,} bytes)"
                )

        except Exception as e:
            result.error = str(e)
            logger.error(f"Failed to download {source.title}: {e}")

        return result

    def download_notebook_sources(self, notebook_id: str,
                                   output_dir: str,
                                   progress_callback=None,
                                   timeout: float = 30.0
                                   ) -> List[NLMSourceDownloadResult]:
        """
        Download all source files from a notebook.

        Args:
            notebook_id: The notebook to download sources from.
            output_dir: Directory to save files (created if needed).
            progress_callback: Optional callback(msg) for progress updates.
            timeout: Per-file HTTP timeout in seconds.

        Returns:
            List of NLMSourceDownloadResult for each source.
        """
        emit = progress_callback or (lambda msg: None)

        sources = self.list_sources(notebook_id)
        emit(f"Found {len(sources)} sources")

        results = []
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for i, src in enumerate(sources, 1):
            emit(f"  [{i}/{len(sources)}] {src.title}")

            if not src.url:
                result = NLMSourceDownloadResult(
                    source_id=src.id,
                    title=src.title,
                    error="No URL (uploaded file — cannot download)",
                )
                results.append(result)
                emit(f"    ⏭️ Skipped — no URL")
                continue

            dl_result = self.download_source_file(src, output_dir, timeout)
            results.append(dl_result)

            if dl_result.success:
                size_str = f"{dl_result.size_bytes / 1024:.0f}KB"
                emit(f"    ✅ Downloaded ({size_str})")
            else:
                emit(f"    ❌ {dl_result.error}")

        return results
