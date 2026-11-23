"""
Backend-neutral interfaces for Gemini Notebook (formerly NotebookLM).

Two roles, deliberately separated:

* ``NotebookWriter`` — create notebooks and add sources. Implemented by the
  official Gemini Notebook API (``official.py``).
* ``NotebookReader`` — read things back (notes, extracted text). The official
  API cannot do this yet ("Retrieving information from a notebook" is on
  Google's roadmap), so today it is implemented only by the opt-in
  experimental adapter (``experimental.py``). When Google ships read access,
  a new reader slots in here without touching the sync engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Protocol, runtime_checkable


class SourceState(str, Enum):
    PROCESSING = "PROCESSING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    DELETING = "DELETING"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def parse(cls, value: Optional[str]) -> "SourceState":
        try:
            return cls(str(value).upper())
        except ValueError:
            return cls.UNKNOWN

    @property
    def is_terminal(self) -> bool:
        return self in (SourceState.ACTIVE, SourceState.FAILED)


# google.rpc.Code values the API documents for FAILED sources → plain English.
FAILURE_REASONS = {
    3: "Rejected: too long, empty, from a blocked site, or blocked by safety filters.",
    7: "The page is behind a paywall or login.",
    8: "This notebook has reached its source limit for your plan.",
    9: "The source could not be reached.",
    13: "Internal error while processing (YouTube videos with no transcript land here).",
}


def explain_failure(code: Optional[int], message: str = "") -> str:
    if code in FAILURE_REASONS:
        return FAILURE_REASONS[code]
    return message or "Processing failed for an unknown reason."


@dataclass
class Notebook:
    name: str                      # "notebooks/{uuid}"
    display_name: str
    create_time: str = ""

    @property
    def id(self) -> str:
        return self.name.rsplit("/", 1)[-1]

    @property
    def url(self) -> str:
        return f"https://notebook.google.com/notebook/{self.id}"


@dataclass
class Source:
    name: str                      # "notebooks/{nb}/sources/{src}"
    display_name: str = ""
    state: SourceState = SourceState.UNKNOWN
    mime_type: str = ""
    original_url: Optional[str] = None
    create_time: str = ""
    error_code: Optional[int] = None
    error_message: str = ""

    @property
    def id(self) -> str:
        return self.name.rsplit("/", 1)[-1]

    @property
    def notebook_name(self) -> str:
        return self.name.split("/sources/", 1)[0]

    @property
    def failure_reason(self) -> str:
        return explain_failure(self.error_code, self.error_message)


@dataclass
class Note:
    id: str
    title: str
    content: str = ""


@dataclass
class SourceText:
    source_id: str
    title: str
    content: str = ""


@runtime_checkable
class NotebookWriter(Protocol):
    def list_notebooks(self) -> List[Notebook]: ...
    def get_notebook(self, notebook: str) -> Notebook: ...
    def create_notebook(self, display_name: str) -> Notebook: ...
    def rename_notebook(self, notebook: str, display_name: str) -> Notebook: ...
    def list_sources(self, notebook: str) -> List[Source]: ...
    def get_source(self, source_name: str) -> Source: ...
    def upload_file(self, notebook: str, path: Path, display_name: str,
                    mime_type: Optional[str] = None) -> Source: ...
    def add_url(self, notebook: str, url: str,
                display_name: Optional[str] = None) -> Source: ...
    def add_text(self, notebook: str, display_name: str, content: str,
                 mime_type: str = "text/markdown") -> Source: ...
    def wait_for(self, source_names: List[str], timeout: float = 900,
                 on_update=None) -> Dict[str, Source]: ...


@runtime_checkable
class NotebookReader(Protocol):
    def list_notes(self, notebook_id: str) -> List[Note]: ...
    def get_source_text(self, notebook_id: str, source_id: str) -> SourceText: ...


def notebook_name(notebook: str) -> str:
    """Accept either a bare id or 'notebooks/{id}' and return the resource name."""
    return notebook if notebook.startswith("notebooks/") else f"notebooks/{notebook}"
