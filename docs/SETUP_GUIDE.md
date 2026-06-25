# CiteBridge — Setup & Implementation Guide

**One-button bidirectional sync between Zotero and Google NotebookLM.**

This guide takes you from a blank machine to a working sync. It's layered: a
**fast path** for confident users up top, then a **detailed walkthrough** and a
deep **troubleshooting** section below. If you only read one thing, read the box
directly below — it covers the three things that trip almost everyone up.

---

## ⚠️ Read this first — the 3 things that trip people up

1. **You need Python 3.10 or newer.** Older versions (3.9 and below) will fail to
   install the NotebookLM library. Check with `python3 --version`.

2. **The NotebookLM login needs a browser engine (Playwright + Chromium).** This
   is the #1 reason setup fails partway through. The setup script installs it for
   you, but if you install manually you must run `playwright install chromium`
   **before** `notebooklm login`. (See [Step 3](#step-3--authenticate-notebooklm).)

3. **Your NotebookLM login expires every few weeks.** This is normal — it uses
   your Google account cookies, which Google rotates. When sync suddenly stops
   working with an "authentication expired" message, the fix is almost always to
   **re-run `notebooklm login`**. Nothing is broken.

> **Why is NotebookLM finicky?** Google has no public NotebookLM API for personal
> accounts (only an Enterprise/Google Cloud one). CiteBridge uses a community
> library that drives NotebookLM the same way your browser does. It works well,
> but it depends on your Google session staying valid. Your **Zotero** side, by
> contrast, uses an official, stable API and rarely needs attention.

---

## Fast path (≈15 minutes)

For users comfortable with a terminal. Everyone else: skip to the
[detailed walkthrough](#detailed-walkthrough).

```bash
# 1. Get the code and enter the folder
git clone https://github.com/RobVoss-AI/Citebridge.git
cd Citebridge

# 2. One-command setup (creates venv, installs deps + browser, runs login)
chmod +x setup.sh && ./setup.sh        # macOS / Linux
#   setup.bat                          # Windows: double-click instead

# 3. Launch the app
source .venv/bin/activate               # macOS / Linux  (.venv\Scripts\activate on Windows)
streamlit run app.py
```

Then in the browser tab that opens (usually `http://localhost:8501`):

1. Sidebar → **Zotero Connection** → paste your API Key + Library ID → **Save & Test Zotero** (green check).
2. Sidebar → **NotebookLM Connection** → **Verify NotebookLM Connection** (status flips to ✅ Connected).
3. Main page → **Push to NotebookLM** tab → check a collection → **PUSH TO NOTEBOOKLM**.

Need a Zotero API key? See [Step 4](#step-4--get-your-zotero-api-key--library-id).

---

## Detailed walkthrough

### Step 0 — Prerequisites

| You need | How to check / get it |
|----------|------------------------|
| **Python 3.10+** | Run `python3 --version`. If it's missing or older than 3.10, install from [python.org/downloads](https://www.python.org/downloads/). |
| **Zotero**, with at least one collection of items | The free [Zotero desktop app](https://www.zotero.org/download/). Add a few PDFs to a collection to test with. |
| **A Google account** with NotebookLM access | Confirm you can open [notebooklm.google.com](https://notebooklm.google.com) and see your notebooks. |
| **A terminal** | macOS: *Terminal* app. Windows: *Command Prompt* or *PowerShell*. |

### Step 1 — Get the code

```bash
git clone https://github.com/RobVoss-AI/Citebridge.git
cd Citebridge
```

No Git? Download the ZIP from the GitHub page (green **Code** button → **Download ZIP**),
unzip it, and `cd` into the folder.

### Step 2 — Set up the environment

**Option A — automatic (recommended).** The setup script creates an isolated
Python environment, installs everything (including the browser engine), and
walks you through login:

```bash
chmod +x setup.sh && ./setup.sh     # macOS / Linux
```

On **Windows**, double-click `setup.bat` instead.

**Option B — manual.** If you'd rather run each step yourself:

```bash
python3 -m venv .venv               # create an isolated environment
source .venv/bin/activate           # macOS / Linux
#   .venv\Scripts\activate          # Windows

pip install -r requirements.txt     # install CiteBridge's dependencies
playwright install chromium         # one-time browser download for NotebookLM login
```

> **What's a virtual environment (`.venv`)?** A sandbox that keeps CiteBridge's
> dependencies from colliding with other Python tools on your machine. You
> **activate** it (`source .venv/bin/activate`) each time you open a new terminal
> to work with CiteBridge. Your prompt will show `(.venv)` when it's active.

### Step 3 — Authenticate NotebookLM

```bash
notebooklm login
```

This opens a Chromium browser window. **Sign in to your Google account**, wait
until you see the NotebookLM page load, then return to the terminal and **press
Enter** to save your session.

- Your session is stored locally at `~/.notebooklm/storage_state.json`. It never
  leaves your machine.
- **If you see `Playwright not installed`:** you skipped the browser step. Run
  `pip install "notebooklm-py[browser]"` then `playwright install chromium`, and
  try again. (The automatic setup does this for you.)
- **Remember:** this session expires every few weeks. Re-running `notebooklm login`
  is the standard fix when sync later fails.

### Step 4 — Get your Zotero API key + Library ID

1. Go to [zotero.org/settings/keys](https://www.zotero.org/settings/keys).
2. Click **Create new private key**.
3. Name it **CiteBridge**.
4. Under *Personal Library*, check **Allow library access** and **Allow write
   access** (write is needed to sync notes back into Zotero).
5. Click **Save Key** and **copy the key** — you won't be able to see it again.
6. Note your **Library ID**: the number shown near the top of that page
   ("Your userID for use in API calls is **XXXXXXX**"). It's a *number*, not your
   username.

### Step 5 — Launch the app

```bash
source .venv/bin/activate     # if not already active (.venv\Scripts\activate on Windows)
streamlit run app.py
```

A browser tab opens at `http://localhost:8501`. Leave the terminal running while
you use CiteBridge — closing it stops the app.

### Step 6 — Connect Zotero

In the sidebar, expand **🔶 Zotero Connection**:

1. Paste your **API Key** and **Library ID**.
2. Leave **Library Type** as `user` (use `group` only for shared group libraries).
3. *(Optional but faster)* Set **Local Storage Path** to your `Zotero/storage`
   folder so CiteBridge reads PDFs from disk instead of downloading them.
   CiteBridge tries to auto-detect this.
4. Click **💾 Save & Test Zotero** — you should get a green ✅.

### Step 7 — Verify NotebookLM

Expand **🟣 NotebookLM Connection** and click **🔗 Verify NotebookLM Connection**.

CiteBridge does a **real** check here, so the status it reports is trustworthy:

| Status on the dashboard | What it means | What to do |
|--------------------------|---------------|------------|
| ✅ **Connected** | A live API call succeeded this session. | You're good — start syncing. |
| 🔑 **Verify needed** | Login credentials exist, but haven't been confirmed this session. | Click **Verify NotebookLM Connection**. |
| ❌ **Not authenticated** | No login found. | Run `notebooklm login` in your terminal (Step 3). |

> If **Verify** fails even though you logged in earlier, your session has almost
> certainly **expired** — re-run `notebooklm login` and verify again.

### Step 8 — Your first push (Zotero → NotebookLM)

1. Open the **📤 Push to NotebookLM** tab.
2. Check one or more collections.
3. Click **📤 PUSH TO NOTEBOOKLM**.

CiteBridge creates a NotebookLM notebook named after each collection and uploads
its PDFs as sources. NotebookLM then needs a little time to process each PDF —
larger files take longer. Already-synced items are skipped automatically on
future runs.

### Step 9 — The other workflows

CiteBridge has four tabs:

- **📤 Push to NotebookLM** — Zotero collections → NotebookLM notebooks (Step 8).
- **📥 Import Sources to Zotero** — pull a notebook's sources *into* Zotero as
  proper library items, with URLs, item types, and the **full extracted text**
  saved as a note. Each notebook becomes a Zotero collection prefixed `NLM:`.
- **💾 Download Source Files** — download the original source files (PDFs, web
  pages) from a notebook to a local folder. Great for sources NotebookLM's
  research features added that you don't have a copy of.
- **📋 Sync History** — a log of past runs.

**Reverse note sync:** under **🔄 Sync Options** you can enable *"Sync NotebookLM
notes back to Zotero."* When on, AI-generated notes from a notebook are matched
to Zotero items by title and attached as notes tagged `notebooklm-sync`.

---

## How it works (and why your data is safe)

```
Zotero Collection          Google NotebookLM
┌─────────────────┐        ┌──────────────────┐
│  AI Ethics      │──push─►│  AI Ethics       │
│  ├── paper1.pdf │        │  ├── paper1 (src) │
│  ├── paper2.pdf │◄─import┤  ├── paper2 (src) │
│  └── paper3.pdf │◄─notes─┤  └── AI Notes     │
└─────────────────┘        └──────────────────┘
```

- **State tracking.** CiteBridge keeps a small local database at
  `~/.citebridge/sync_state.db` recording what's already synced (with file
  hashes), so repeat runs only handle new or changed items.
- **Your settings** live at `~/.citebridge/config.yaml`. Your credentials never
  leave your machine.
- **Data safety.** CiteBridge only *reads* from Zotero, except when you opt in to
  writing notes back. It never deletes or modifies your existing Zotero items. If
  the NotebookLM connection ever breaks, your Zotero library is untouched.

---

## Troubleshooting

Messages below are the actual ones you'll see, with the fix.

### "Authentication expired or invalid. Redirected to accounts.google.com…"
Your NotebookLM session expired (this happens every few weeks). **Fix:**
```bash
source .venv/bin/activate
notebooklm login
```
Then click **Verify NotebookLM Connection** in the app.

### Dashboard shows 🔑 "Verify needed" and won't say Connected
Click **Verify NotebookLM Connection** in the sidebar. If it then errors, your
session expired — re-run `notebooklm login`.

### "Playwright not installed. Run: pip install notebooklm[browser] …"
You skipped the browser engine. **Fix:**
```bash
pip install "notebooklm-py[browser]"
playwright install chromium
```
Then retry `notebooklm login`.

### `notebooklm: command not found`
Your virtual environment isn't active. **Fix:** `source .venv/bin/activate`
(macOS/Linux) or `.venv\Scripts\activate` (Windows), then try again.

### `ModuleNotFoundError: No module named 'streamlit'` (or `pyzotero`, etc.)
Same cause — the venv isn't active, or dependencies weren't installed. Activate
the venv and run `pip install -r requirements.txt`.

### NotebookLM library won't install / weird install errors
You're likely on Python older than 3.10. Check `python3 --version` and install a
newer Python, then recreate the venv.

### "Zotero connection failed"
- Double-check the **API key** (re-copy it from the Zotero keys page).
- The **Library ID is a number**, not your username.
- Make sure the key has **library access** enabled (and **write access** if you
  want note sync-back).

### "No PDFs found for items"
Some items simply have no PDF attachment. For items that do, set the **Local
Storage Path** to your `Zotero/storage` folder, or let CiteBridge fall back to
downloading via the Zotero API.

### Streamlit didn't open / "port already in use"
Run it on a different port: `streamlit run app.py --server.port 8502`.

### Sync seems slow
Normal — NotebookLM processes each uploaded PDF before the next step, and large
PDFs (50 MB+) take longer.

### Don't blindly upgrade the NotebookLM library
CiteBridge pins `notebooklm-py` to a tested version. Newer releases change the
API and can break the import flow. Avoid `pip install --upgrade notebooklm-py`
unless you intend to re-test.

---

## Maintenance & FAQ

- **How often do I re-authenticate NotebookLM?** Whenever it stops working —
  typically every few weeks. It's a 30-second `notebooklm login`.
- **Do I have to keep the terminal open?** Yes, while the app is running. Closing
  it stops the local server.
- **Where is my data stored?** Settings: `~/.citebridge/config.yaml`. Sync state:
  `~/.citebridge/sync_state.db`. NotebookLM session: `~/.notebooklm/`. All local.
- **Can I run this on a server / Streamlit Cloud?** Yes — set the
  `NOTEBOOKLM_AUTH_JSON` secret to the contents of your
  `~/.notebooklm/storage_state.json`. (Cloud sessions still expire and need
  refreshing.)
- **Is my Zotero data ever at risk?** No. CiteBridge only writes to Zotero when
  you explicitly enable note sync-back, and even then it only *adds* notes.

---

## Quick command reference

```bash
# First-time setup (manual)
python3 -m venv .venv
source .venv/bin/activate            # .venv\Scripts\activate on Windows
pip install -r requirements.txt
playwright install chromium
notebooklm login

# Every time you use CiteBridge
source .venv/bin/activate
streamlit run app.py

# When NotebookLM stops working
notebooklm login                     # then click "Verify" in the app
```

---

*Built by [Voss AI Consulting](https://www.vossaiconsulting.com). Questions or
custom integrations: [vossaiconsulting.com](https://www.vossaiconsulting.com).*
