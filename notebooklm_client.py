"""
NotebookLM API client — wraps notebooklm-py for creating notebooks,
adding sources, and reading notes.

Uses the unofficial notebooklm-py library (async API).
All public methods in this wrapper are synchronous for easy integration.
"""

import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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
    url: Optional[str] = None  # Original URL (web pages, YouTube, etc.)


@dataclass
class NLMSourceFull:
    """A source with its full extracted text content."""
    id: str
    title: str
    source_type: str = ""
    url: Optional[str] = None
    content: str = ""  # Full text extracted by NotebookLM
    char_count: int = 0


@dataclass
class NLMNote:
    """Represents a note in a NotebookLM notebook."""
    id: str
    title: str
    content: str = ""


def _run_async(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an existing event loop (e.g., Streamlit)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


class NotebookLMClient:
    """
    High-level synchronous client for NotebookLM.
    Wraps the async notebooklm-py library.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the NotebookLM client.

        Auth tokens are loaded from storage (created by `notebooklm login`).

        Args:
            storage_path: Optional path to storage_state.json.
                          If None, uses default ~/.notebooklm/ location.
        """
        self._client = None
        self._storage_path = storage_path
        self._initialized = False

    def _ensure_client(self):
        """Lazily initialize the async client."""
        if self._initialized:
            return

        try:
            from notebooklm.auth import load_auth_from_storage, fetch_tokens, get_storage_path, AuthTokens
            from notebooklm.client import NotebookLMClient as _AsyncClient

            path = Path(self._storage_path) if self._storage_path else get_storage_path()
            if not path.exists():
                raise FileNotFoundError(
                    f"NotebookLM auth not found at {path}. "
                    "Run 'notebooklm login' first to authenticate."
                )

            # Load cookies from storage
            cookies = load_auth_from_storage(path)
            # Fetch CSRF and session tokens
            csrf_token, session_id = _run_async(self._async_fetch_tokens(cookies))

            auth = AuthTokens(
                cookies=cookies,
                csrf_token=csrf_token,
                session_id=session_id,
            )
            self._client = _AsyncClient(auth=auth)
            self._initialized = True
            logger.info("NotebookLM client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize NotebookLM client: {e}")
            raise

    @staticmethod
    async def _async_fetch_tokens(cookies):
        from notebooklm.auth import fetch_tokens
        return await asyncio.to_thread(fetch_tokens, cookies)

    def test_connection(self) -> bool:
        """Test if we can connect to NotebookLM."""
        try:
            self._ensure_client()
            notebooks = self.list_notebooks()
            logger.info(f"NotebookLM connection OK — {len(notebooks)} notebooks found")
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
            # Find the notebooklm CLI
            cli_path = "notebooklm"
            # Check common locations
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
                capture_output=False,  # Let user interact with browser
                timeout=300,  # 5 minute timeout
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False

    @staticmethod
    def is_authenticated() -> bool:
        """Check if NotebookLM auth tokens exist."""
        try:
            from notebooklm.auth import get_storage_path
            return get_storage_path().exists()
        except Exception:
            return False

    # ── Notebook Operations ──

    def list_notebooks(self) -> List[NLMNotebook]:
        """List all notebooks in the account."""
        self._ensure_client()

        async def _list():
            return await self._client.notebooks.list()

        raw = _run_async(_list())
        notebooks = []
        for nb in raw:
            notebooks.append(NLMNotebook(
                id=nb.id if hasattr(nb, "id") else str(nb),
                title=nb.title if hasattr(nb, "title") else "Untitled",
                sources_count=nb.sources_count if hasattr(nb, "sources_count") else 0,
                created_at=str(nb.created_at) if hasattr(nb, "created_at") else "",
            ))
        return notebooks

    def create_notebook(self, title: str) -> NLMNotebook:
        """Create a new notebook."""
        self._ensure_client()

        async def _create():
            return await self._client.notebooks.create(title)

        nb = _run_async(_create())
        result = NLMNotebook(
            id=nb.id if hasattr(nb, "id") else str(nb),
            title=nb.title if hasattr(nb, "title") else title,
        )
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

    # ── Source Operations ──

    def add_pdf_source(self, notebook_id: str, file_path: str,
                       wait: bool = True, timeout: float = 120.0) -> NLMSource:
        """
        Upload a PDF file as a source to a notebook.

        Args:
            notebook_id: The notebook to add the source to
            file_path: Local path to the PDF file
            wait: If True, wait for source to finish processing
            timeout: Max seconds to wait for processing
        """
        self._ensure_client()

        async def _add():
            return await self._client.sources.add_file(
                notebook_id, file_path,
                wait=wait, wait_timeout=timeout,
            )

        src = _run_async(_add())
        result = NLMSource(
            id=src.id if hasattr(src, "id") else str(src),
            title=src.title if hasattr(src, "title") else Path(file_path).stem,
            source_type=str(src.source_type) if hasattr(src, "source_type") else "pdf",
            status=str(src.status) if hasattr(src, "status") else "",
            is_ready=src.is_ready if hasattr(src, "is_ready") else False,
        )
        logger.info(f"Added source: {result.title} → notebook {notebook_id}")
        return result

    def add_url_source(self, notebook_id: str, url: str,
                       wait: bool = True) -> NLMSource:
        """Add a URL as a source to a notebook."""
        self._ensure_client()

        async def _add():
            return await self._client.sources.add_url(
                notebook_id, url, wait=wait,
            )

        src = _run_async(_add())
        return NLMSource(
            id=src.id if hasattr(src, "id") else str(src),
            title=src.title if hasattr(src, "title") else url,
            source_type="web_page",
            is_ready=src.is_ready if hasattr(src, "is_ready") else False,
        )

    def list_sources(self, notebook_id: str) -> List[NLMSource]:
        """List all sources in a notebook."""
        self._ensure_client()

        async def _list():
            return await self._client.sources.list(notebook_id)

        raw = _run_async(_list())
        return [
            NLMSource(
                id=s.id if hasattr(s, "id") else str(s),
                title=s.title if hasattr(s, "title") else "Untitled",
                source_type=str(s.source_type) if hasattr(s, "source_type") else "",
                status=str(s.status) if hasattr(s, "status") else "",
                is_ready=s.is_ready if hasattr(s, "is_ready") else False,
                url=s.url if hasattr(s, "url") else None,
            )
            for s in raw
        ]

    def get_source_fulltext(self, notebook_id: str,
                             source_id: str) -> NLMSourceFull:
        """
        Get the full extracted text content of a source.
        This is the key method for pulling sources into Zotero —
        it returns everything NotebookLM extracted from the original document.
        """
        self._ensure_client()

        async def _get():
            return await self._client.sources.get_fulltext(notebook_id, source_id)

        ft = _run_async(_get())
        return NLMSourceFull(
            id=ft.source_id if hasattr(ft, "source_id") else source_id,
            title=ft.title if hasattr(ft, "title") else "",
            source_type=str(ft.source_type) if hasattr(ft, "source_type") else "",
            url=ft.url if hasattr(ft, "url") else None,
            content=ft.content if hasattr(ft, "content") else "",
            char_count=ft.char_count if hasattr(ft, "char_count") else 0,
        )

    def get_source_guide(self, notebook_id: str,
                          source_id: str) -> Dict[str, Any]:
        """Get the AI-generated study guide for a source."""
        self._ensure_client()

        async def _get():
            return await self._client.sources.get_guide(notebook_id, source_id)

        return _run_async(_get())

    def get_all_sources_with_content(self, notebook_id: str
                                      ) -> List[NLMSourceFull]:
        """
        Get all sources in a notebook with their full text content.
        This is the primary method for importing a NotebookLM research
        collection into Zotero.
        """
        sources = self.list_sources(notebook_id)
        results = []

        for src in sources:
            try:
                full = self.get_source_fulltext(notebook_id, src.id)
                # Merge the URL from list_sources if fulltext didn't have it
                if not full.url and src.url:
                    full.url = src.url
                if not full.source_type and src.source_type:
                    full.source_type = src.source_type
                results.append(full)
                logger.info(f"Got fulltext for: {full.title} ({full.char_count} chars)")
            except Exception as e:
                logger.error(f"Failed to get fulltext for {src.title}: {e}")
                # Still include with basic info
                results.append(NLMSourceFull(
                    id=src.id, title=src.title,
                    source_type=src.source_type, url=src.url,
                ))

        return results

    # ── Note Operations ──

    def list_notes(self, notebook_id: str) -> List[NLMNote]:
        """List all notes in a notebook."""
        self._ensure_client()

        async def _list():
            return await self._client.notes.list(notebook_id)

        raw = _run_async(_list())
        return [
            NLMNote(
                id=n.id if hasattr(n, "id") else str(n),
                title=n.title if hasattr(n, "title") else "Untitled",
                content=n.content if hasattr(n, "content") else "",
            )
            for n in raw
        ]

    def get_notebook_summary(self, notebook_id: str) -> str:
        """Get AI-generated summary for a notebook."""
        self._ensure_client()

        async def _summary():
            return await self._client.notebooks.get_summary(notebook_id)

        return _run_async(_summary())
