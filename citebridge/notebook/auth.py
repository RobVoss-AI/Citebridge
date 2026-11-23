"""
Google sign-in for the Gemini Notebook API.

Uses the OAuth "installed app" (desktop) flow: a browser opens, the user signs
in to their *personal* Google account (Workspace accounts aren't supported by
the API yet), and we keep a refresh token so they don't sign in again.

The OAuth client (client_id/secret) belongs to the CiteBridge publisher and is
loaded from, in order:
  1. $CITEBRIDGE_OAUTH_CLIENT  — a path to client JSON, or the JSON itself
  2. $CITEBRIDGE_HOME/oauth_client.json
  3. citebridge/oauth_client.json bundled with a release (git-ignored)

Tokens are stored per Google account behind the TokenStore interface: a 0600
file today, swappable for a database/KMS store in a hosted deployment.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Callable, List, Optional, Protocol

from ..config import citebridge_home

logger = logging.getLogger(__name__)

NOTEBOOK_SCOPE = "https://www.googleapis.com/auth/notebook"
SCOPES = [NOTEBOOK_SCOPE, "openid", "https://www.googleapis.com/auth/userinfo.email"]

_BUNDLED_CLIENT = Path(__file__).resolve().parent.parent / "oauth_client.json"


class AuthError(Exception):
    """Sign-in is missing or no longer valid; the user must sign in again."""


class OAuthClientMissing(AuthError):
    pass


# ── OAuth client config ──

def load_client_config() -> dict:
    env = os.environ.get("CITEBRIDGE_OAUTH_CLIENT", "").strip()
    if env:
        if env.startswith("{"):
            return json.loads(env)
        return json.loads(Path(env).expanduser().read_text())
    for path in (citebridge_home() / "oauth_client.json", _BUNDLED_CLIENT):
        if path.exists():
            return json.loads(path.read_text())
    raise OAuthClientMissing(
        "No Google OAuth client found. Put the CiteBridge oauth_client.json in "
        f"{citebridge_home()} (or set CITEBRIDGE_OAUTH_CLIENT).")


def has_client_config() -> bool:
    try:
        load_client_config()
        return True
    except Exception:
        return False


def client_project_id(config: Optional[dict] = None) -> str:
    try:
        config = config or load_client_config()
        return (config.get("installed") or config.get("web") or {}).get("project_id", "")
    except Exception:
        return ""


# ── token storage ──

class TokenStore(Protocol):
    def load(self, account: str) -> Optional[str]: ...        # authorized-user JSON
    def save(self, account: str, token_json: str) -> None: ...
    def delete(self, account: str) -> None: ...
    def accounts(self) -> List[str]: ...


class FileTokenStore:
    """One 0600 JSON file per account under $CITEBRIDGE_HOME/accounts/."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or citebridge_home() / "accounts")

    @staticmethod
    def _slug(account: str) -> str:
        return re.sub(r"[^A-Za-z0-9._@-]", "_", account)

    def _path(self, account: str) -> Path:
        return self.base_dir / self._slug(account) / "token.json"

    def load(self, account: str) -> Optional[str]:
        p = self._path(account)
        return p.read_text() if p.exists() else None

    def save(self, account: str, token_json: str) -> None:
        p = self._path(account)
        p.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(p.parent, 0o700)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(token_json)

    def delete(self, account: str) -> None:
        p = self._path(account)
        if p.exists():
            p.unlink()

    def accounts(self) -> List[str]:
        if not self.base_dir.exists():
            return []
        return sorted(d.name for d in self.base_dir.iterdir() if (d / "token.json").exists())


# ── identity ──

def _account_from_id_token(id_token: Optional[str]) -> Optional[str]:
    """Pull the email (fallback: sub) out of the ID token we just received from Google.

    The token came straight from Google's token endpoint over TLS during the
    code exchange, so we only need its claims, not a signature check.
    """
    if not id_token:
        return None
    try:
        payload = id_token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        return claims.get("email") or claims.get("sub")
    except Exception:
        return None


# ── sign-in / credentials ──

def sign_in(store: Optional[TokenStore] = None, open_browser: bool = True) -> str:
    """Run the browser OAuth flow; store the token; return the account id (email)."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    store = store or FileTokenStore()
    flow = InstalledAppFlow.from_client_config(load_client_config(), SCOPES)
    creds = flow.run_local_server(
        port=0, open_browser=open_browser, prompt="consent",
        authorization_prompt_message="Opening your browser to sign in to Google…",
        success_message="Signed in. You can close this tab and return to CiteBridge.")
    account = _account_from_id_token(getattr(creds, "id_token", None)) or "default"
    granted = getattr(creds, "granted_scopes", None) or creds.scopes or []
    if NOTEBOOK_SCOPE not in granted:
        raise AuthError("Gemini Notebook access wasn't granted. Please sign in again and "
                        "allow CiteBridge to manage your notebooks.")
    store.save(account, creds.to_json())
    logger.info(f"Signed in as {account}")
    return account


def sign_out(account: str, store: Optional[TokenStore] = None) -> None:
    (store or FileTokenStore()).delete(account)


def load_credentials(account: str, store: Optional[TokenStore] = None):
    """Return valid google.oauth2 Credentials, refreshing (and persisting) if needed."""
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    store = store or FileTokenStore()
    raw = store.load(account)
    if not raw:
        raise AuthError("Not signed in to Google.")
    creds = Credentials.from_authorized_user_info(json.loads(raw), SCOPES)
    if not creds.valid:
        if not creds.refresh_token:
            raise AuthError("Your Google sign-in has expired. Please sign in again.")
        try:
            creds.refresh(Request())
        except RefreshError as e:
            # Most commonly: the OAuth app is still in "Testing" mode, where
            # Google expires refresh tokens after 7 days, or access was revoked.
            raise AuthError("Your Google sign-in has expired or was revoked. "
                            "Please sign in again.") from e
        store.save(account, creds.to_json())
    return creds


def token_provider(account: str, store: Optional[TokenStore] = None
                   ) -> Callable[[bool], str]:
    """A callable returning a bearer token; pass force=True after a 401 to refresh."""
    store = store or FileTokenStore()
    state = {"creds": None}

    def get(force: bool = False) -> str:
        if force:
            # Drop the cached access token so load_credentials() refreshes it.
            raw = store.load(account)
            if raw:
                data = json.loads(raw)
                data.pop("token", None)
                data.pop("expiry", None)
                store.save(account, json.dumps(data))
            state["creds"] = None
        creds = state["creds"]
        if creds is None or not creds.valid:
            creds = load_credentials(account, store)
            state["creds"] = creds
        return creds.token

    return get


def signed_in_accounts(store: Optional[TokenStore] = None) -> List[str]:
    return (store or FileTokenStore()).accounts()
