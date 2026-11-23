"""Official Gemini Notebook API client, against httpx.MockTransport."""

import json

import httpx
import pytest

from citebridge.notebook.base import SourceState
from citebridge.notebook.official import ApiError, OfficialNotebookClient, is_supported_mime
from citebridge.ratelimit import ApiLimits

UPLOAD_URL = "https://notebook.googleapis.com/upload/session/xyz"


def client(handler, tokens=None):
    tokens = tokens if tokens is not None else []

    def token(force=False):
        tokens.append(force)
        return "fresh" if force else "tok"
    sleeps = []
    c = OfficialNotebookClient(token, http=httpx.Client(transport=httpx.MockTransport(handler)),
                               limits=ApiLimits(sleep=lambda s: None), sleep=sleeps.append)
    return c, sleeps


def err(status, grpc, msg, details=()):
    return httpx.Response(status, json={"error": {"code": status, "status": grpc, "message": msg,
                                                  "details": list(details)}})


def test_create_notebook_sends_documented_snake_case_and_parses_camel_case():
    seen = {}

    def h(req):
        seen.update(path=req.url.path, body=json.loads(req.content), auth=req.headers["authorization"])
        return httpx.Response(200, json={"name": "notebooks/nb1", "displayName": "Q3",
                                         "createTime": "2026-08-06T04:00:00Z"})
    nb = client(h)[0].create_notebook("Q3")
    assert seen == {"path": "/v1/notebooks", "body": {"display_name": "Q3"}, "auth": "Bearer tok"}
    assert (nb.id, nb.display_name) == ("nb1", "Q3")


def test_text_and_url_sources_use_documented_bodies():
    bodies = []

    def h(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(200, json={"name": "notebooks/nb1/sources/s1", "state": "PROCESSING"})
    c = client(h)[0]
    c.add_text("nb1", "Card", "# hi")
    c.add_url("notebooks/nb1", "https://example.org", "Smith (2020) — X")
    assert bodies[0] == {"display_name": "Card", "mime_type": "text/markdown",
                         "text_content": {"content": "# hi"}}
    assert bodies[1] == {"web_content": {"url": "https://example.org"},
                         "display_name": "Smith (2020) — X"}
    with pytest.raises(ValueError):
        c.add_url("nb1", "http://insecure.example")
    with pytest.raises(ValueError):
        c.add_text("nb1", "big", "x" * 1_000_001)


def test_resumable_upload_handshake(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 hello")
    calls = []

    def h(req):
        calls.append(req)
        if req.url.path.endswith("sources:upload"):
            return httpx.Response(200, headers={"X-Goog-Upload-URL": UPLOAD_URL})
        return httpx.Response(200, json={"source": {"name": "notebooks/nb1/sources/s9",
                                                    "displayName": "Smith (2023) — Paper",
                                                    "mimeType": "application/pdf",
                                                    "state": "PROCESSING"}})
    src = client(h)[0].upload_file("nb1", pdf, "Smith (2023) — Paper")
    start, finalize = calls
    assert start.url.path == "/upload/v1/notebooks/nb1/sources:upload"
    assert start.headers["x-goog-upload-protocol"] == "resumable"
    assert start.headers["x-goog-upload-command"] == "start"
    assert start.headers["x-goog-upload-header-content-length"] == str(len(b"%PDF-1.4 hello"))
    assert start.headers["x-goog-upload-header-content-type"] == "application/pdf"
    assert json.loads(start.content) == {"source": {"displayName": "Smith (2023) — Paper"}}
    assert str(finalize.url) == UPLOAD_URL
    assert finalize.headers["x-goog-upload-command"] == "upload, finalize"
    assert finalize.headers["x-goog-upload-offset"] == "0"
    assert "authorization" not in finalize.headers            # docs: no token on 2nd request
    assert finalize.content == b"%PDF-1.4 hello"
    assert (src.name, src.state) == ("notebooks/nb1/sources/s9", SourceState.PROCESSING)


def test_upload_restarts_after_429_using_retry_delay(tmp_path):
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF")
    starts = []

    def h(req):
        if req.url.path.endswith("sources:upload"):
            starts.append(1)
            if len(starts) == 1:
                return err(429, "RESOURCE_EXHAUSTED", "slow down",
                           [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "3s"}])
            return httpx.Response(200, headers={"X-Goog-Upload-URL": UPLOAD_URL})
        return httpx.Response(200, json={"source": {"name": "notebooks/n/sources/s"}})
    c, sleeps = client(h)
    c.upload_file("n", pdf, "x")
    assert len(starts) == 2
    assert 3.0 <= sleeps[0] <= 3.5                            # honoured retryDelay


def test_rejects_oversize_and_unsupported_files(tmp_path):
    exe = tmp_path / "tool.exe"
    exe.write_bytes(b"MZ")
    with pytest.raises(ValueError, match="isn't supported"):
        client(lambda r: httpx.Response(500))[0].upload_file("n", exe, "x")
    assert is_supported_mime("image/png") and is_supported_mime("audio/mpeg")
    assert not is_supported_mime("application/zip")


def test_pagination_and_owner_filter():
    pages = {None: {"notebooks": [{"name": "notebooks/a"}], "nextPageToken": "p2"},
             "p2": {"notebooks": [{"name": "notebooks/b"}]}}
    filters = []

    def h(req):
        filters.append(req.url.params.get("filter"))
        assert req.url.params["pageSize"] == "100"
        return httpx.Response(200, json=pages[req.url.params.get("pageToken")])
    nbs = client(h)[0].list_notebooks()
    assert [n.id for n in nbs] == ["a", "b"]
    assert filters[0] == "notebook.is_owner = true"


def test_401_refreshes_token_once():
    tokens, calls = [], []

    def h(req):
        calls.append(req.headers["authorization"])
        if len(calls) == 1:
            return httpx.Response(401, json={"error": {"message": "expired"}})
        return httpx.Response(200, json={"name": "notebooks/a"})
    client(h, tokens)[0].get_notebook("a")
    assert calls == ["Bearer tok", "Bearer fresh"]


def test_failed_source_and_friendly_errors():
    def h(req):
        if req.url.path.endswith("/sources/bad"):
            return httpx.Response(200, json={"name": "notebooks/n/sources/bad", "state": "FAILED",
                                             "error": {"code": 7, "message": "paywall"}})
        if req.url.path == "/v1/notebooks" and req.method == "POST":
            return err(404, "NOT_FOUND", "Account not found for user.")
        return err(403, "PERMISSION_DENIED", "nope", [{
            "@type": "type.googleapis.com/google.rpc.ErrorInfo",
            "reason": "WORKSPACE_ACCOUNT_NOT_SUPPORTED"}])
    c = client(h)[0]
    src = c.get_source("notebooks/n/sources/bad")
    assert src.state == SourceState.FAILED and "paywall" in src.failure_reason.lower()
    with pytest.raises(ApiError, match="notebook.google.com once"):
        c.create_notebook("x")
    with pytest.raises(ApiError, match="personal Google account"):
        c.list_notebooks()


def test_wait_for_polls_until_terminal():
    states = iter(["PROCESSING", "PROCESSING", "ACTIVE"])

    def h(req):
        return httpx.Response(200, json={"name": "notebooks/n/sources/s", "state": next(states)})
    c, sleeps = client(h)
    done = c.wait_for(["notebooks/n/sources/s"])
    assert done["notebooks/n/sources/s"].state == SourceState.ACTIVE
    assert sleeps == [2.0, 3.0]                                # backoff between polls
