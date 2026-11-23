"""
EXPERIMENTAL read-back from Gemini Notebook via the unofficial ``notebooklm-py``.

The official Gemini Notebook API can't read anything back yet (Google lists
"Retrieving information from a notebook" on its roadmap). Until it can, this
optional adapter implements ``NotebookReader`` using notebooklm-py, which
drives Gemini Notebook with browser cookies from ``notebooklm login``.

Isolation rules:
* This is the ONLY module that touches cookies or notebooklm-py.
* It is read-only — all writes go through the official API.
* If the library or its login is missing, ``ExperimentalReader.available()``
  returns False with a reason, and the rest of CiteBridge carries on.
* When Google ships official read access, add an official reader and delete
  this file; nothing else needs to change.

Tested against notebooklm-py 0.8.6 (see requirements-experimental.txt).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from pathlib import Path
from typing import Any, Callable, List, Tuple

from .base import Note, SourceText

logger = logging.getLogger(__name__)


def _run(coro):
    """Run a coroutine from sync code, even inside an existing loop (Streamlit)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class ExperimentalReader:
    def __init__(self, storage_path: str = ""):
        self.storage_path = storage_path or None

    @classmethod
    def available(cls, storage_path: str = "") -> Tuple[bool, str]:
        try:
            import notebooklm  # noqa: F401
            from notebooklm.paths import get_storage_path
        except Exception:
            return False, ("The experimental add-on isn't installed. Run: "
                           "pip install -r requirements-experimental.txt")
        path = Path(storage_path) if storage_path else get_storage_path()
        if not path.exists():
            return False, "Not logged in. Run `notebooklm login` in your terminal first."
        return True, "Ready"

    def _with_client(self, fn: Callable[[Any], Any]):
        from notebooklm.client import NotebookLMClient

        async def go():
            async with NotebookLMClient.from_storage(self.storage_path) as client:
                return await fn(client)
        return _run(go())

    def check(self) -> Tuple[bool, str]:
        """Live check — cookies expire every few weeks."""
        try:
            n = self._with_client(lambda c: c.notebooks.list())
            return True, f"Connected ({len(n)} notebooks visible)"
        except Exception as e:
            msg = str(e)
            if "expired" in msg.lower() or "accounts.google.com" in msg:
                return False, "Experimental login expired. Run `notebooklm login` again."
            return False, f"Experimental read-back failed: {msg[:200]}"

    # ── NotebookReader ──

    def list_notes(self, notebook_id: str) -> List[Note]:
        async def op(c):
            return await c.notes.list(notebook_id)
        return [Note(id=n.id, title=n.title or "Untitled note", content=n.content or "")
                for n in self._with_client(op)]

    def get_source_text(self, notebook_id: str, source_id: str) -> SourceText:
        async def op(c):
            return await c.sources.get_fulltext(notebook_id, source_id, output_format="markdown")
        ft = self._with_client(op)
        return SourceText(source_id=source_id, title=ft.title or "", content=ft.content or "")
