# CiteBridge v2 — Setup Guide

Connect your Zotero library to **Gemini Notebook** (formerly NotebookLM). This takes about
10 minutes. The fast path is first; the full walkthrough and troubleshooting follow.

---

## ⚠️ Read this first

1. **Use a personal Google account.** Gemini Notebook's API doesn't support work or school
   (Google Workspace) accounts yet.
2. **Open Gemini Notebook once** at [notebook.google.com](https://notebook.google.com) with that
   account before your first sync, so your account exists there.
3. **You need Python 3.10 or newer.** Check with `python3 --version`.

---

## Fast path

```bash
git clone https://github.com/RobVoss-AI/Citebridge.git
cd Citebridge
# Put oauth_client.json (from your CiteBridge download) in this folder, then:
chmod +x setup.sh && ./setup.sh        # Windows: double-click setup.bat
source .venv/bin/activate              # Windows: .venv\Scripts\activate
streamlit run app.py
```

In the CiteBridge tab that opens:
1. Click **Sign in with Google** and approve access to your notebooks.
2. Open **Zotero 7 desktop** (CiteBridge reads your library from it).
3. Tick your collections → **Preview what will be sent** → **Send to Gemini Notebook**.

---

## Detailed walkthrough

### 1. Install

Run `setup.sh` (macOS/Linux) or `setup.bat` (Windows). It creates a private Python environment,
installs CiteBridge, and puts the Google sign-in file (`oauth_client.json`) in place. When it asks
about the **experimental add-on**, answer **N** unless you want notes pulled back into Zotero (see step 7).

Manual install, if you prefer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p ~/.citebridge && cp oauth_client.json ~/.citebridge/
```

### 2. Let CiteBridge read Zotero

Open **Zotero 7** → **Settings → Advanced** → tick **Allow other applications on this computer to
communicate with Zotero**. Keep Zotero open while you sync. No API key is needed to *send* to
Gemini Notebook.

> No Zotero desktop on this machine? Add a Zotero API key instead (step 6) and CiteBridge reads
> through zotero.org.

### 3. Sign in with Google

`streamlit run app.py`, then **Sign in with Google** in the sidebar. A browser tab asks you to allow
CiteBridge to manage your Gemini Notebook notebooks. CiteBridge stores the sign-in on your computer
only (`~/.citebridge/accounts/`), so you won't need to repeat this.

> While CiteBridge is in Google's early-access review, Google may show an "unverified app" screen
> (choose **Advanced → Go to CiteBridge**) and ask you to sign in again about once a week.

### 4. Pick your plan size

Sidebar → **Sync settings** → choose your Gemini Notebook plan (Free, Plus, Pro, Ultra). This sets how
many sources fit in one notebook (50 on Free). Collections that don't fit are split into
*Name (1/2)*, *Name (2/2)*… automatically.

Other options (all on by default):
- **Bibliography source** — lists every item with author-year, DOI and Zotero key, so Gemini cites
  your sources correctly. Recommended.
- **My highlights & notes** — your PDF annotations and Zotero notes, gathered into one source.
- **Items without a PDF** — sent as a web page (blog posts, videos, news) or as a citation card
  (everything else).

### 5. Send collections

Tick collections → **Preview what will be sent**. The preview shows exactly what will be created
before anything happens. Then **Send to Gemini Notebook**. Sources appear in your notebook named
**Author (Year) — Title**. The **Processing** tab shows anything still being processed or rejected,
with the reason in plain English.

Running it again only sends what's new or changed.

### 6. Bring research back into Zotero

Add your Zotero API key first: [zotero.org/settings/keys](https://www.zotero.org/settings/keys) →
**Create new private key** → allow library access **and write access** → paste the key and your
numeric **Library ID** into the sidebar's Zotero section.

Then on the **Bring back to Zotero** tab:
- **Import sources I added in Gemini Notebook** — web pages, videos and Discover results you added
  there become Zotero items in a "From Gemini Notebook: …" collection.

### 7. (Optional) Experimental: pull notes and text back

Gemini Notebook's official API can't read notes or source text yet. The experimental add-on can, using
an unofficial library with its own browser login that **expires every few weeks**:

```bash
pip install -r requirements-experimental.txt
playwright install chromium
notebooklm login
```

Then enable it in the sidebar (🧪 Experimental). **Pull notes into Zotero** attaches each note to the
item it cites by author-year; notes citing several sources (or none) are saved together as one note
on the collection, never guessed.

---

## Cleanup: why sources pile up

Gemini Notebook's API can't delete yet. When you replace a PDF or remove an item from a collection,
the old source stays in the notebook (and uses one of its source slots). The **Cleanup** tab lists
them with links. Delete them in Gemini Notebook and click **Done**, or just sync again; CiteBridge
notices they're gone.

---

## Troubleshooting

| You see | Do this |
|---|---|
| "Google's sign-in file is missing" | Copy `oauth_client.json` from your CiteBridge download to `~/.citebridge/`. |
| "Work/school accounts aren't supported" | Sign out and sign in with a personal Gmail account. |
| "This Google account hasn't used Gemini Notebook yet" | Open [notebook.google.com](https://notebook.google.com) once with that account, then retry. |
| "Your Google sign-in has expired" | Click **Sign in with Google** again. (Weekly during early access.) |
| "Can't read your Zotero library" | Open Zotero 7 and enable the setting in step 2, or add an API key. |
| A source **FAILED: paywall / unreachable** | Web pages that fail are automatically replaced by a citation card. |
| A source **FAILED: source limit** | Your notebook is full. Clean up old sources, or set the right plan size. |
| "Experimental login expired" | Run `notebooklm login` again (only affects the experimental add-on). |

---

*Built by [Voss AI Consulting](https://www.vossaiconsulting.com). Want it set up for you?
[Book a call](https://meetings-na2.hubspot.com/rob-voss).*
