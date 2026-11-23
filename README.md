# CiteBridge

**Zotero ↔ Gemini Notebook (formerly NotebookLM), built for accurate research.**

CiteBridge sends your Zotero collections into Gemini Notebook as organized
notebooks — every source named and traceable back to the exact Zotero item —
and brings research you find in Gemini Notebook back into Zotero.

Built by [Voss AI Consulting](https://www.vossaiconsulting.com)

> 📘 **Installing it?** Read the [Setup Guide](docs/SETUP_GUIDE.md).
> **Publishing it?** See [Publisher setup](docs/PUBLISHER_SETUP.md).

---

## What's new in v2

Google renamed NotebookLM to **Gemini Notebook** (July 2026) and released the
official **Gemini Notebook API**. CiteBridge v2 is rebuilt on it:

| | v1 (2026) | v2 |
|---|---|---|
| Sign-in | Browser cookies; expired every few weeks | **Sign in with Google** (OAuth); stays signed in |
| Install | Needed Playwright + Chromium | Plain Python packages |
| Zotero | Web API key required | Reads straight from **Zotero 7 desktop**; key only needed to write back |
| Items without a PDF | Skipped | Sent as a web page or a **citation card** |
| Accuracy | Sources named by title; notes matched by fuzzy title | Sources named **Author (Year) — Title**; a **Bibliography** source per notebook; every source mapped to its Zotero key |
| Big collections | Silently hit the source limit | Split across notebooks to fit your plan |
| Speed | Waited on each upload | Uploads all, then checks processing together |

## What it does

**Zotero → Gemini Notebook** (official API)
- Each collection becomes a notebook (split into *Name (1/2)*, *(2/2)*… if it exceeds your plan's source limit).
- PDFs upload directly. Web pages and YouTube go in as web sources. Everything else goes in as a Markdown
  **citation card** with the full reference, abstract and your Zotero notes.
- Each notebook also gets a **📚 Bibliography** (every item with author-year, DOI and Zotero key) and
  optionally **🖍 Highlights & notes** (your PDF annotations).
- Re-running only sends what's new or changed.

**Gemini Notebook → Zotero**
- **Import sources you added in Gemini Notebook** (web pages, YouTube, Discover results) as Zotero items — official API.
- **Pull notes** into Zotero — *experimental add-on* (see below). Notes attach to an item only when they
  unambiguously cite it; everything else goes into one collection note. Never guessed.
- **Download originals** behind web sources.

## Honest limits (Gemini Notebook API, Public Preview)

- **Personal Google accounts only** — work/school (Workspace) accounts aren't supported yet.
- **Write-only for now.** The API can create notebooks and add sources, but can't yet read notes,
  chat, citations or extracted text ("Retrieving information" is on Google's roadmap). Until it can,
  reading back uses the **optional experimental add-on** built on the unofficial `notebooklm-py`, whose
  cookie login expires every few weeks. It's off by default and isolated so it can be swapped for the
  official API the day it ships.
- **No deletes yet.** Replaced or removed sources are listed in the **Cleanup** tab for you to delete in
  Gemini Notebook; CiteBridge notices when you have.
- **Limits:** 20 uploads/min per user; 200 MB per file; 50–600 sources per notebook depending on your plan.

## Architecture

```
app.py                         Streamlit UI
citebridge/
  config.py                    settings (~/.citebridge/config.yaml)
  state.py                     SQLite state, per Google account (source ↔ Zotero item mapping, orphans)
  ratelimit.py                 token buckets + 429 RetryInfo backoff
  notebook/
    base.py                    NotebookWriter / NotebookReader interfaces
    auth.py                    Google sign-in (OAuth installed-app flow), per-account token store
    official.py                Gemini Notebook API v1 client (resumable uploads, polling)
    experimental.py            optional read-back via notebooklm-py
  zotero/
    client.py                  Zotero 7 local API + web API
    render.py                  display names, citation cards, bibliography, highlights digest
  sync/
    planner.py                 what to send (cap-aware, sticky, change detection) — pure logic
    executor.py                send, poll, fall back, adopt v1 notebooks
    reverse.py                 import sources, pull notes, download originals
scripts/smoke_official.py      live check against the real API
tests/                         pytest suite (fake API + fake Zotero)
```

Development: `pip install -r requirements.txt -r requirements-dev.txt && pytest`

## Author

**Rob Voss, Ph.D.** — [Voss AI Consulting](https://www.vossaiconsulting.com) · [robvoss.com](https://www.robvoss.com)

## License

MIT — see [LICENSE](LICENSE). CiteBridge is an independent tool, not affiliated with Zotero or Google.
Built with [pyzotero](https://github.com/urschrei/pyzotero); the experimental add-on uses
[notebooklm-py](https://github.com/teng-lin/notebooklm-py).
