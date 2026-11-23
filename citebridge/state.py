"""
SQLite sync state, schema v2.

Every row is scoped to a Google ``account`` so one database can serve several
signed-in users (and so a hosted deployment is a storage swap, not a rewrite).

Key idea for accuracy: ``sources`` maps every Gemini Notebook source resource
name to the Zotero item it came from. Reverse sync and citation tracing use
this mapping — never fuzzy title matching.

The API cannot delete sources, so anything that should go away (a replaced
PDF, an item removed from Zotero) is recorded in ``orphans`` for the user to
clean up by hand — or automatically once Google ships delete.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

SCHEMA_VERSION = 2

# Source kinds
PDF, FILE, URL, CARD, BIBLIOGRAPHY, ANNOTATIONS = (
    "pdf", "file", "url", "card", "bibliography", "annotations")
COLLECTION_LEVEL_KINDS = (BIBLIOGRAPHY, ANNOTATIONS)

_V2_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS notebooks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account TEXT NOT NULL,
    collection_key TEXT NOT NULL,
    part INTEGER NOT NULL DEFAULT 1,
    notebook_name TEXT NOT NULL,
    display_name TEXT,
    created_at TEXT,
    last_synced TEXT,
    UNIQUE(account, collection_key, part)
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account TEXT NOT NULL,
    collection_key TEXT NOT NULL,
    notebook_name TEXT NOT NULL,
    source_name TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    zotero_key TEXT,
    display_name TEXT,
    content_hash TEXT,
    zotero_version INTEGER,
    state TEXT,
    error_code INTEGER,
    error_message TEXT,
    superseded INTEGER NOT NULL DEFAULT 0,
    created_at TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_sources_item ON sources(account, collection_key, zotero_key);

CREATE TABLE IF NOT EXISTS orphans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account TEXT NOT NULL,
    source_name TEXT NOT NULL UNIQUE,
    notebook_name TEXT,
    display_name TEXT,
    reason TEXT,
    detected_at TEXT,
    resolved INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS note_imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account TEXT NOT NULL,
    notebook_id TEXT NOT NULL,
    note_id TEXT NOT NULL,
    zotero_item_key TEXT,
    zotero_note_key TEXT,
    imported_at TEXT,
    UNIQUE(account, note_id)
);

CREATE TABLE IF NOT EXISTS source_imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account TEXT NOT NULL,
    source_name TEXT NOT NULL,
    zotero_item_key TEXT,
    imported_at TEXT,
    UNIQUE(account, source_name)
);

CREATE TABLE IF NOT EXISTS sync_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    account TEXT,
    action TEXT NOT NULL,
    status TEXT NOT NULL,
    details TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class NotebookRecord:
    collection_key: str
    part: int
    notebook_name: str
    display_name: str

    @property
    def notebook_id(self) -> str:
        return self.notebook_name.rsplit("/", 1)[-1]


@dataclass
class SourceRecord:
    collection_key: str
    notebook_name: str
    source_name: str
    kind: str
    zotero_key: Optional[str]
    display_name: str
    content_hash: Optional[str]
    zotero_version: Optional[int]
    state: Optional[str]
    error_code: Optional[int]
    error_message: Optional[str]
    superseded: bool

    @property
    def source_id(self) -> str:
        return self.source_name.rsplit("/", 1)[-1]


@dataclass
class Orphan:
    source_name: str
    notebook_name: str
    display_name: str
    reason: str
    detected_at: str

    @property
    def notebook_url(self) -> str:
        return f"https://notebook.google.com/notebook/{self.notebook_name.rsplit('/', 1)[-1]}"


@dataclass
class LegacyCollection:
    """A v1 collection↔notebook mapping awaiting adoption by the v2 engine."""
    collection_key: str
    collection_name: str
    notebook_id: str


class SyncState:
    def __init__(self, db_path: Optional[str] = None, account: str = ""):
        if db_path is None:
            from .config import db_file
            db_path = str(db_file())
        self.db_path = db_path
        self.account = account
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def for_account(self, account: str) -> "SyncState":
        return SyncState(self.db_path, account)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ── schema / migration ──

    def _migrate(self) -> None:
        with self._conn() as c:
            tables = {r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            version = None
            if "schema_meta" in tables:
                row = c.execute("SELECT value FROM schema_meta WHERE key='version'").fetchone()
                version = int(row[0]) if row else None
            if version == SCHEMA_VERSION:
                return

            # v1 → v2: keep the old tables (renamed) so their notebooks can be adopted.
            for old in ("synced_collections", "synced_items", "nlm_notes_synced"):
                if old in tables and f"legacy_{old}" not in tables:
                    c.execute(f"ALTER TABLE {old} RENAME TO legacy_{old}")
            if "sync_log" in tables:
                cols = {r[1] for r in c.execute("PRAGMA table_info(sync_log)")}
                if "account" not in cols:
                    c.execute("ALTER TABLE sync_log ADD COLUMN account TEXT")

            c.executescript(_V2_SCHEMA)

            if "nlm_notes_synced" in tables:  # carry over "already imported" markers
                c.execute("""INSERT OR IGNORE INTO note_imports
                             (account, notebook_id, note_id, zotero_item_key, zotero_note_key, imported_at)
                             SELECT '', nlm_notebook_id, nlm_note_id, zotero_item_key,
                                    zotero_note_key, synced_at FROM legacy_nlm_notes_synced""")
            c.execute("INSERT OR REPLACE INTO schema_meta VALUES ('version', ?)",
                      (str(SCHEMA_VERSION),))

    # ── legacy adoption ──

    def legacy_collections(self) -> List[LegacyCollection]:
        with self._conn() as c:
            has = c.execute("SELECT 1 FROM sqlite_master WHERE name='legacy_synced_collections'").fetchone()
            if not has:
                return []
            rows = c.execute("""SELECT zotero_key, zotero_name, nlm_notebook_id
                                FROM legacy_synced_collections
                                WHERE nlm_notebook_id IS NOT NULL AND nlm_notebook_id != ''""").fetchall()
        return [LegacyCollection(*r) for r in rows]

    def forget_legacy_collection(self, collection_key: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM legacy_synced_collections WHERE zotero_key = ?", (collection_key,))

    # ── notebooks ──

    def notebooks_for(self, collection_key: str) -> List[NotebookRecord]:
        with self._conn() as c:
            rows = c.execute("""SELECT collection_key, part, notebook_name, display_name
                                FROM notebooks WHERE account=? AND collection_key=?
                                ORDER BY part""", (self.account, collection_key)).fetchall()
        return [NotebookRecord(*r) for r in rows]

    def all_notebooks(self) -> List[NotebookRecord]:
        with self._conn() as c:
            rows = c.execute("""SELECT collection_key, part, notebook_name, display_name
                                FROM notebooks WHERE account=? ORDER BY collection_key, part""",
                             (self.account,)).fetchall()
        return [NotebookRecord(*r) for r in rows]

    def upsert_notebook(self, collection_key: str, part: int, notebook_name: str,
                        display_name: str) -> None:
        now = _now()
        with self._conn() as c:
            c.execute("""INSERT INTO notebooks
                         (account, collection_key, part, notebook_name, display_name, created_at, last_synced)
                         VALUES (?, ?, ?, ?, ?, ?, ?)
                         ON CONFLICT(account, collection_key, part) DO UPDATE SET
                           notebook_name=excluded.notebook_name,
                           display_name=excluded.display_name,
                           last_synced=excluded.last_synced""",
                      (self.account, collection_key, part, notebook_name, display_name, now, now))

    def touch_notebook(self, notebook_name: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE notebooks SET last_synced=? WHERE account=? AND notebook_name=?",
                      (_now(), self.account, notebook_name))

    # ── sources ──

    _SRC_COLS = ("collection_key, notebook_name, source_name, kind, zotero_key, display_name, "
                 "content_hash, zotero_version, state, error_code, error_message, superseded")

    def _sources(self, where: str, args: tuple) -> List[SourceRecord]:
        with self._conn() as c:
            rows = c.execute(f"SELECT {self._SRC_COLS} FROM sources WHERE account=? AND {where} "
                             f"ORDER BY id", (self.account, *args)).fetchall()
        return [SourceRecord(*r[:-1], bool(r[-1])) for r in rows]

    def record_source(self, *, collection_key: str, notebook_name: str, source_name: str,
                      kind: str, display_name: str, zotero_key: Optional[str] = None,
                      content_hash: Optional[str] = None, zotero_version: Optional[int] = None,
                      state: str = "PROCESSING") -> None:
        now = _now()
        with self._conn() as c:
            c.execute("""INSERT INTO sources
                         (account, collection_key, notebook_name, source_name, kind, zotero_key,
                          display_name, content_hash, zotero_version, state, created_at, updated_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                         ON CONFLICT(source_name) DO UPDATE SET
                           state=excluded.state, updated_at=excluded.updated_at""",
                      (self.account, collection_key, notebook_name, source_name, kind, zotero_key,
                       display_name, content_hash, zotero_version, state, now, now))

    def update_source_state(self, source_name: str, state: str,
                            error_code: Optional[int] = None, error_message: str = "") -> None:
        with self._conn() as c:
            c.execute("""UPDATE sources SET state=?, error_code=?, error_message=?, updated_at=?
                         WHERE account=? AND source_name=?""",
                      (state, error_code, error_message or None, _now(), self.account, source_name))

    def sources_for_collection(self, collection_key: str,
                               include_superseded: bool = False) -> List[SourceRecord]:
        where = "collection_key=?" + ("" if include_superseded else " AND superseded=0")
        return self._sources(where, (collection_key,))

    def sources_in_notebook(self, notebook_name: str) -> List[SourceRecord]:
        return self._sources("notebook_name=? AND superseded=0", (notebook_name,))

    def display_names_in_notebook(self, notebook_name: str) -> set:
        """Every display name CiteBridge has ever used in a notebook (incl. replaced ones)."""
        with self._conn() as c:
            rows = c.execute("SELECT display_name FROM sources WHERE account=? AND notebook_name=?",
                             (self.account, notebook_name)).fetchall()
        return {r[0] for r in rows if r[0]}

    def current_source_for_item(self, collection_key: str, zotero_key: str) -> Optional[SourceRecord]:
        recs = self._sources("collection_key=? AND zotero_key=? AND superseded=0",
                             (collection_key, zotero_key))
        return recs[-1] if recs else None

    def collection_source(self, collection_key: str, notebook_name: str,
                          kind: str) -> Optional[SourceRecord]:
        recs = self._sources("collection_key=? AND notebook_name=? AND kind=? AND superseded=0",
                             (collection_key, notebook_name, kind))
        return recs[-1] if recs else None

    def pending_sources(self) -> List[SourceRecord]:
        return self._sources("superseded=0 AND (state IS NULL OR state IN ('PROCESSING','UNKNOWN'))", ())

    def failed_sources(self) -> List[SourceRecord]:
        return self._sources("superseded=0 AND state='FAILED'", ())

    def item_key_for_source(self, source_id_or_name: str) -> Optional[str]:
        sid = source_id_or_name.rsplit("/", 1)[-1]
        with self._conn() as c:
            row = c.execute("""SELECT zotero_key FROM sources
                               WHERE account=? AND (source_name=? OR source_name LIKE ?)
                               ORDER BY id DESC LIMIT 1""",
                            (self.account, source_id_or_name, f"%/sources/{sid}")).fetchone()
        return row[0] if row else None

    def supersede(self, source_name: str, reason: str) -> None:
        """Retire a source and list it for manual cleanup (the API can't delete)."""
        with self._conn() as c:
            row = c.execute("""SELECT notebook_name, display_name FROM sources
                               WHERE account=? AND source_name=?""",
                            (self.account, source_name)).fetchone()
            c.execute("UPDATE sources SET superseded=1, updated_at=? WHERE account=? AND source_name=?",
                      (_now(), self.account, source_name))
            if row:
                c.execute("""INSERT OR IGNORE INTO orphans
                             (account, source_name, notebook_name, display_name, reason, detected_at)
                             VALUES (?, ?, ?, ?, ?, ?)""",
                          (self.account, source_name, row[0], row[1], reason, _now()))

    # ── orphans ──

    def orphans(self, include_resolved: bool = False) -> List[Orphan]:
        where = "" if include_resolved else " AND resolved=0"
        with self._conn() as c:
            rows = c.execute(f"""SELECT source_name, notebook_name, display_name, reason, detected_at
                                 FROM orphans WHERE account=?{where} ORDER BY detected_at""",
                             (self.account,)).fetchall()
        return [Orphan(*r) for r in rows]

    def resolve_orphan(self, source_name: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE orphans SET resolved=1 WHERE account=? AND source_name=?",
                      (self.account, source_name))

    # ── reverse sync / imports ──

    def note_imported(self, note_id: str) -> bool:
        with self._conn() as c:
            row = c.execute("SELECT 1 FROM note_imports WHERE note_id=? AND account IN (?, '')",
                            (note_id, self.account)).fetchone()
        return row is not None

    def record_note_import(self, notebook_id: str, note_id: str,
                           zotero_item_key: str = "", zotero_note_key: str = "") -> None:
        with self._conn() as c:
            c.execute("""INSERT OR REPLACE INTO note_imports
                         (account, notebook_id, note_id, zotero_item_key, zotero_note_key, imported_at)
                         VALUES (?, ?, ?, ?, ?, ?)""",
                      (self.account, notebook_id, note_id, zotero_item_key, zotero_note_key, _now()))

    def source_imported(self, source_name: str) -> Optional[str]:
        with self._conn() as c:
            row = c.execute("SELECT zotero_item_key FROM source_imports WHERE account=? AND source_name=?",
                            (self.account, source_name)).fetchone()
        return row[0] if row else None

    def record_source_import(self, source_name: str, zotero_item_key: str) -> None:
        with self._conn() as c:
            c.execute("""INSERT OR REPLACE INTO source_imports
                         (account, source_name, zotero_item_key, imported_at) VALUES (?, ?, ?, ?)""",
                      (self.account, source_name, zotero_item_key, _now()))

    # ── log & stats ──

    def log(self, action: str, status: str, details: str = "") -> None:
        with self._conn() as c:
            c.execute("INSERT INTO sync_log (account, action, status, details) VALUES (?, ?, ?, ?)",
                      (self.account, action, status, details))

    def recent_logs(self, limit: int = 50) -> List[tuple]:
        with self._conn() as c:
            return c.execute("""SELECT timestamp, action, status, details FROM sync_log
                                WHERE account=? OR account IS NULL OR account=''
                                ORDER BY id DESC LIMIT ?""", (self.account, limit)).fetchall()

    def stats(self) -> dict:
        with self._conn() as c:
            one = lambda q: c.execute(q, (self.account,)).fetchone()[0]  # noqa: E731
            return {
                "notebooks": one("SELECT COUNT(*) FROM notebooks WHERE account=?"),
                "sources": one("SELECT COUNT(*) FROM sources WHERE account=? AND superseded=0"),
                "processing": one("SELECT COUNT(*) FROM sources WHERE account=? AND superseded=0 "
                                  "AND state IN ('PROCESSING','UNKNOWN')"),
                "failed": one("SELECT COUNT(*) FROM sources WHERE account=? AND superseded=0 "
                              "AND state='FAILED'"),
                "orphans": one("SELECT COUNT(*) FROM orphans WHERE account=? AND resolved=0"),
                "last_sync": one("SELECT MAX(last_synced) FROM notebooks WHERE account=?") or "Never",
            }
