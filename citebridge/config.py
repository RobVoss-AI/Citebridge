"""
Configuration for CiteBridge v2.

Everything lives under CITEBRIDGE_HOME (default ~/.citebridge). Overriding the
home dir keeps tests isolated and lets a future hosted deployment give each
tenant its own directory.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)

# (max notebooks per user, max sources per notebook) — from the API rate-limits page.
TIER_LIMITS: Dict[str, tuple] = {
    "free": (100, 50),
    "plus": (200, 100),
    "pro": (500, 300),
    "ultra": (500, 600),
}


def citebridge_home() -> Path:
    return Path(os.environ.get("CITEBRIDGE_HOME", Path.home() / ".citebridge")).expanduser()


def config_file() -> Path:
    return citebridge_home() / "config.yaml"


def db_file() -> Path:
    return citebridge_home() / "sync_state.db"


@dataclass
class ZoteroConfig:
    api_key: str = ""             # only needed to WRITE to Zotero (notes back, imports)
    library_id: str = ""
    library_type: str = "user"    # "user" | "group"
    local_storage_path: str = ""  # Zotero/storage for direct PDF reads
    use_local_api: bool = True    # read via Zotero 7 desktop (localhost:23119) when running


@dataclass
class GoogleConfig:
    account: str = ""             # active Google account id (email) — see notebook.auth
    quota_project: str = ""       # optional x-goog-user-project override


@dataclass
class SyncConfig:
    enabled_collections: List[str] = field(default_factory=list)
    plan_tier: str = "free"                 # sets the sources-per-notebook cap
    include_bibliography: bool = True       # one "Bibliography" source per notebook
    include_annotations: bool = True        # one "Highlights & notes" source per notebook
    cite_items_without_pdf: bool = True     # URL / citation-card sources for non-PDF items
    max_file_size_mb: int = 200             # API hard limit is 200 MiB
    source_cap_override: int = 0            # >0: use this cap instead (e.g. leave headroom)

    @property
    def sources_per_notebook(self) -> int:
        if self.source_cap_override > 0:
            return self.source_cap_override
        return TIER_LIMITS.get(self.plan_tier, TIER_LIMITS["free"])[1]


@dataclass
class ExperimentalConfig:
    enabled: bool = False         # unofficial read-back via notebooklm-py
    storage_path: str = ""        # notebooklm-py storage_state.json (blank = default)


def _build(cls, data: Dict[str, Any]):
    """Construct a dataclass from a dict, ignoring keys it doesn't know (old configs)."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (data or {}).items() if k in known})


@dataclass
class AppConfig:
    zotero: ZoteroConfig = field(default_factory=ZoteroConfig)
    google: GoogleConfig = field(default_factory=GoogleConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    experimental: ExperimentalConfig = field(default_factory=ExperimentalConfig)

    def save(self) -> None:
        path = config_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, default_flow_style=False, sort_keys=False)
        os.chmod(path, 0o600)  # contains the Zotero API key

    @classmethod
    def load(cls) -> "AppConfig":
        path = config_file()
        if not path.exists():
            return cls()
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except Exception as e:  # corrupt file — don't crash the app
            logger.error(f"Could not read {path}: {e}")
            return cls()

        cfg = cls(
            zotero=_build(ZoteroConfig, data.get("zotero")),
            google=_build(GoogleConfig, data.get("google")),
            sync=_build(SyncConfig, data.get("sync")),
            experimental=_build(ExperimentalConfig, data.get("experimental")),
        )
        # v1 → v2: the old "notebooklm" section configured the cookie-based
        # client, which now lives behind the experimental toggle.
        legacy = data.get("notebooklm") or {}
        if legacy.get("storage_path") and not cfg.experimental.storage_path:
            cfg.experimental.storage_path = legacy["storage_path"]
        return cfg

    def is_zotero_readable(self) -> bool:
        return self.zotero.use_local_api or bool(self.zotero.api_key and self.zotero.library_id)

    def is_zotero_writable(self) -> bool:
        return bool(self.zotero.api_key and self.zotero.library_id)
