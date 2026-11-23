"""
Client for the official Gemini Notebook API (notebook.googleapis.com/v1).

Implements ``NotebookWriter``. Notes on following the docs exactly:

* Request bodies use snake_case as documented (``display_name``, ``web_content``,
  ``text_content``, top-level ``mime_type``). Responses come back camelCase;
  we read both spellings defensively.
* File uploads use the documented *resumable* protocol over raw HTTP. The
  Google Python client library can't set a source's display name on upload
  (it becomes "Untitled.pdf"), and display names are how researchers recognise
  sources, so we don't use it.
* The second (finalize) upload request must NOT carry the access token.
* 429s carry google.rpc.RetryInfo; for uploads the docs say to restart the
  upload after ``retryDelay``.
"""

from __future__ import annotations

import logging
import mimetypes
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import httpx

from ..ratelimit import ApiLimits, RetryableError, TokenBucket, parse_duration, with_retries
from .base import Notebook, Source, SourceState, notebook_name

logger = logging.getLogger(__name__)

BASE_URL = "https://notebook.googleapis.com"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024        # 200 MiB
MAX_TEXT_BYTES = 1_000_000                  # text_content limit
MAX_DISPLAY_NAME = 500

_EXTRA_TYPES = {
    ".md": "text/markdown", ".markdown": "text/markdown", ".epub": "application/epub+zip",
    ".hwp": "application/x-hwp", ".odt": "application/vnd.oasis.opendocument.text",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}
_SUPPORTED_EXACT = {
    "application/pdf", "application/epub+zip", "application/rtf", "text/rtf",
    "application/vnd.oasis.opendocument.text", "text/csv", "text/plain", "text/markdown",
    "application/x-hwp", "application/x-hwp-v5", "application/haansofthwp", "application/hwp+zip",
    "application/x-ogg", "video/3gpp", "video/3gpp2", _EXTRA_TYPES[".docx"], _EXTRA_TYPES[".pptx"],
}


def guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    return _EXTRA_TYPES.get(ext) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def is_supported_mime(mime: str) -> bool:
    mime = (mime or "").split(";")[0].strip().lower()
    return mime in _SUPPORTED_EXACT or mime.startswith(("audio/", "image/"))


class ApiError(Exception):
    def __init__(self, status: int, message: str, reason: str = "", grpc_status: str = ""):
        self.status = status
        self.reason = reason
        self.grpc_status = grpc_status
        self.raw_message = message
        super().__init__(self.friendly())

    def friendly(self) -> str:
        msg = self.raw_message or ""
        if self.reason == "WORKSPACE_ACCOUNT_NOT_SUPPORTED":
            return ("Google Workspace (work/school) accounts aren't supported by the Gemini "
                    "Notebook API yet. Please sign in with a personal Google account.")
        if self.status == 404 and "account not found" in msg.lower():
            return ("This Google account hasn't used Gemini Notebook yet. Open "
                    "https://notebook.google.com once with this account, then try again.")
        if self.reason in ("SERVICE_DISABLED", "ACCESS_TOKEN_SCOPE_INSUFFICIENT") or \
                "has not been used in project" in msg:
            return ("CiteBridge's Google project isn't enabled for the Gemini Notebook API, or "
                    "access wasn't granted. Sign out and sign in again; if it persists, contact "
                    f"the CiteBridge publisher. ({msg})")
        return f"Gemini Notebook API error {self.status}: {msg or self.grpc_status}"


class ServerError(RetryableError):
    """5xx: the request MAY have been committed. Only safe to retry idempotent calls."""


def _parse_error(resp: httpx.Response) -> tuple:
    """(message, reason, grpc_status, retry_after) from a google.rpc error body."""
    try:
        err = resp.json().get("error", {})
    except Exception:
        return resp.text[:300], "", "", None
    reason, retry_after = "", None
    for d in err.get("details", []) or []:
        t = d.get("@type", "")
        if t.endswith("RetryInfo"):
            retry_after = parse_duration(d.get("retryDelay"))
        elif t.endswith("ErrorInfo"):
            reason = d.get("reason", "")
    return err.get("message", ""), reason, err.get("status", ""), retry_after


def _g(d: dict, camel: str, snake: str, default=None):
    return d.get(camel, d.get(snake, default))


def parse_notebook(d: dict) -> Notebook:
    return Notebook(name=d["name"], display_name=_g(d, "displayName", "display_name", ""),
                    create_time=_g(d, "createTime", "create_time", ""))


def parse_source(d: dict) -> Source:
    if "source" in d and isinstance(d["source"], dict):   # upload responses are wrapped
        d = d["source"]
    err = d.get("error") or {}
    return Source(
        name=d["name"],
        display_name=_g(d, "displayName", "display_name", ""),
        state=SourceState.parse(d.get("state")),
        mime_type=_g(d, "mimeType", "mime_type", ""),
        original_url=_g(d, "originalUrl", "original_url"),
        create_time=_g(d, "createTime", "create_time", ""),
        error_code=err.get("code"),
        error_message=err.get("message", ""),
    )


class OfficialNotebookClient:
    """Gemini Notebook API v1. ``token`` is a callable(force_refresh: bool) -> str."""

    def __init__(self, token: Callable[[bool], str], *, quota_project: str = "",
                 base_url: str = BASE_URL, http: Optional[httpx.Client] = None,
                 limits: Optional[ApiLimits] = None, sleep: Callable[[float], None] = time.sleep,
                 timeout: float = 120.0):
        self._token = token
        self._quota_project = quota_project
        self._base = base_url.rstrip("/")
        self._http = http or httpx.Client(timeout=timeout)
        self._limits = limits or ApiLimits(sleep=sleep)
        self._sleep = sleep

    # ── transport ──

    def _auth_headers(self, force: bool = False) -> Dict[str, str]:
        h = {"Authorization": f"Bearer {self._token(force)}"}
        if self._quota_project:
            h["x-goog-user-project"] = self._quota_project
        return h

    def _once(self, method: str, url: str, *, bucket: Optional[TokenBucket], auth: bool = True,
              **kw) -> httpx.Response:
        if bucket:
            bucket.acquire()
        headers = dict(kw.pop("headers", {}) or {})
        if auth:
            headers.update(self._auth_headers())
        resp = self._http.request(method, url, headers=headers, **kw)
        if resp.status_code == 401 and auth:
            headers.update(self._auth_headers(force=True))
            resp = self._http.request(method, url, headers=headers, **kw)
        if resp.status_code < 400:
            return resp
        message, reason, grpc, retry_after = _parse_error(resp)
        if resp.status_code == 429:
            raise RetryableError(f"429 {grpc}: {message}", retry_after)
        if resp.status_code >= 500:
            raise ServerError(f"{resp.status_code} {grpc}: {message}", retry_after)
        raise ApiError(resp.status_code, message, reason, grpc)

    def _call(self, method: str, path: str, *, bucket: Optional[TokenBucket] = None,
              idempotent: bool = True, **kw) -> dict:
        def once():
            try:
                return self._once(method, f"{self._base}{path}",
                                  bucket=bucket or self._limits.other, **kw)
            except ServerError as e:
                if idempotent:
                    raise
                # A create may have succeeded server-side; retrying could duplicate a
                # source that the API can't delete. Surface it instead.
                raise ApiError(500, f"{e} — it may or may not have been created; "
                                    "check the notebook before retrying.") from e
        resp = with_retries(
            once,
            sleep=self._sleep,
            on_retry=lambda n, d, e: logger.warning(f"Retry {n} in {d:.1f}s: {e}"))
        return resp.json() if resp.content else {}

    def _paged(self, path: str, key: str, params: Optional[dict] = None) -> Iterable[dict]:
        params = dict(params or {}, pageSize=100)
        while True:
            data = self._call("GET", path, params=params)
            yield from data.get(key, []) or []
            token = data.get("nextPageToken") or data.get("next_page_token")
            if not token:
                return
            params["pageToken"] = token

    # ── notebooks ──

    def list_notebooks(self, owned_only: bool = True) -> List[Notebook]:
        params = {"filter": "notebook.is_owner = true"} if owned_only else {}
        return [parse_notebook(d) for d in self._paged("/v1/notebooks", "notebooks", params)]

    def get_notebook(self, notebook: str) -> Notebook:
        return parse_notebook(self._call("GET", f"/v1/{notebook_name(notebook)}"))

    def create_notebook(self, display_name: str) -> Notebook:
        body = {"display_name": display_name[:MAX_DISPLAY_NAME]}
        return parse_notebook(self._call("POST", "/v1/notebooks", json=body,
                                         bucket=self._limits.other, idempotent=False))

    def rename_notebook(self, notebook: str, display_name: str) -> Notebook:
        name = notebook_name(notebook)
        body = {"name": name, "display_name": display_name[:MAX_DISPLAY_NAME]}
        return parse_notebook(self._call("PATCH", f"/v1/{name}",
                                         params={"updateMask": "displayName"}, json=body))

    # ── sources ──

    def list_sources(self, notebook: str) -> List[Source]:
        return [parse_source(d) for d in
                self._paged(f"/v1/{notebook_name(notebook)}/sources", "sources")]

    def get_source(self, source_name: str) -> Source:
        return parse_source(self._call("GET", f"/v1/{source_name}"))

    def add_url(self, notebook: str, url: str, display_name: Optional[str] = None) -> Source:
        if not url.lower().startswith("https://"):
            raise ValueError("Web sources must be public https:// URLs.")
        body: dict = {"web_content": {"url": url}}
        if display_name:
            body["display_name"] = display_name[:MAX_DISPLAY_NAME]
        return parse_source(self._call("POST", f"/v1/{notebook_name(notebook)}/sources",
                                       json=body, bucket=self._limits.create_source,
                                       idempotent=False))

    def add_text(self, notebook: str, display_name: str, content: str,
                 mime_type: str = "text/markdown") -> Source:
        if len(content.encode("utf-8")) > MAX_TEXT_BYTES:
            raise ValueError("Text sources are limited to 1 MB; split the content first.")
        body = {"display_name": display_name[:MAX_DISPLAY_NAME], "mime_type": mime_type,
                "text_content": {"content": content}}
        return parse_source(self._call("POST", f"/v1/{notebook_name(notebook)}/sources",
                                       json=body, bucket=self._limits.create_source,
                                       idempotent=False))

    def upload_file(self, notebook: str, path: Path, display_name: str,
                    mime_type: Optional[str] = None) -> Source:
        path = Path(path)
        size = path.stat().st_size
        mime = mime_type or guess_mime(path)
        if size > MAX_UPLOAD_BYTES:
            raise ValueError(f"{path.name} is {size / 1048576:.0f} MB; the limit is 200 MB.")
        if not is_supported_mime(mime):
            raise ValueError(f"{path.name}: file type {mime} isn't supported by Gemini Notebook.")
        url = f"{self._base}/upload/v1/{notebook_name(notebook)}/sources:upload"

        def attempt() -> httpx.Response:
            # Per the docs, any 429 means: wait retryDelay, then restart from "start".
            start = self._once("POST", url, bucket=self._limits.upload, headers={
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(size),
                "X-Goog-Upload-Header-Content-Type": mime,
                "Content-Type": "application/json",
            }, json={"source": {"displayName": display_name[:MAX_DISPLAY_NAME]}})
            upload_url = start.headers.get("X-Goog-Upload-URL")
            if not upload_url:
                raise ApiError(start.status_code, "Upload session wasn't created "
                                                  "(no X-Goog-Upload-URL header).")
            with open(path, "rb") as fh:
                try:
                    return self._once("POST", upload_url, bucket=None, auth=False, headers={
                        "X-Goog-Upload-Command": "upload, finalize",
                        "X-Goog-Upload-Offset": "0",
                    }, content=fh.read())
                except ServerError as e:   # may have committed — don't upload twice
                    raise ApiError(500, f"Upload of {path.name} failed ({e}); it may or may "
                                        "not have been added — check the notebook.") from e

        resp = with_retries(attempt, sleep=self._sleep,
                            on_retry=lambda n, d, e: logger.warning(
                                f"Upload {path.name} retry {n} in {d:.1f}s: {e}"))
        src = parse_source(resp.json())
        if not src.display_name:
            src.display_name = display_name
        return src

    # ── processing ──

    def wait_for(self, source_names: List[str], timeout: float = 900,
                 on_update: Optional[Callable[[Source], None]] = None) -> Dict[str, Source]:
        """Poll sources until each is ACTIVE/FAILED or ``timeout`` elapses."""
        pending = list(dict.fromkeys(source_names))
        done: Dict[str, Source] = {}
        interval, deadline = 2.0, time.monotonic() + timeout
        while pending and time.monotonic() < deadline:
            still = []
            for name in pending:
                try:
                    src = self.get_source(name)
                except ApiError as e:
                    if e.status == 404:            # deleted while processing
                        logger.warning(f"{name} disappeared while processing")
                        continue
                    still.append(name)
                    continue
                except Exception as e:             # transient; try again next round
                    logger.warning(f"Status check for {name} failed: {e}")
                    still.append(name)
                    continue
                if src.state.is_terminal:
                    done[name] = src
                    if on_update:
                        on_update(src)
                else:
                    still.append(name)
            pending = still
            if pending:
                self._sleep(interval)
                interval = min(30.0, interval * 1.5)
        return done
