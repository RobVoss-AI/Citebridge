# Publisher setup (Rob only): Google Cloud project & OAuth client

CiteBridge v2 users "Sign in with Google" through **one OAuth client that you own**. This page is
the Phase 0 checklist and the path to Google verification. Nothing here goes in git.

## A. One-time project setup (~15 min)

1. **Create a project** at [console.cloud.google.com](https://console.cloud.google.com) → *New project*
   (e.g. `citebridge`).
2. **Enable the API:** *APIs & Services → Library* → search **Gemini Notebook API**
   (`notebook.googleapis.com`) → *Enable*.
3. **OAuth consent screen** (*Google Auth Platform → Branding / Audience / Data access*):
   - User type **External**; app name **CiteBridge**; support email; homepage
     `https://vossaiconsulting.com/pages/citebridge.html`; privacy policy URL (needed for verification).
   - Scopes: `https://www.googleapis.com/auth/notebook`, `openid`, `.../auth/userinfo.email`.
   - Publishing status **Testing** for now; add yourself (and any pilot users, max 100) as **test users**.
4. **Create the client:** *Clients → Create client → Desktop app* → download the JSON.
5. Save it as `~/.citebridge/oauth_client.json` (`chmod 600`).
6. Sign in once at [notebook.google.com](https://notebook.google.com) with a **personal** Google account.

## B. Run the smoke test

```bash
source .venv/bin/activate
python scripts/smoke_official.py
```

It signs you in, creates/reuses **one** notebook called *CiteBridge smoke test* (the API can't delete
notebooks, so it never makes more), uploads a 1-page PDF and a Markdown note, waits for both to be
**ACTIVE**, and prints the raw JSON. Send me that output; it confirms the field names, upload headers
and response shapes the client was built from the docs, before real users depend on them.

## C. Distributing the client

Bundle `oauth_client.json` **next to** `setup.sh` in the release ZIP you give users (setup copies it
into `~/.citebridge/`), or as a GitHub Release asset. Don't commit it: `.gitignore` already blocks it.
For a desktop OAuth client Google treats the secret as non-confidential, but committing it publicly
trips secret scanning and invites reuse of your quota.

## D. Before inviting real users: Testing-mode limits

While the app is in **Testing**:
- only listed test users (max **100**) can sign in;
- Google expires their sign-in **every 7 days** (users click *Sign in with Google* again).

To lift both, publish the app (*Audience → Publish app*) and complete **Google verification** for
the `notebook` scope: a privacy policy and homepage on vossaiconsulting.com, a short demo video of the
sign-in and what CiteBridge does with notebook access, and possibly a security review depending on
how Google classifies the scope. Start this early; it's the long pole for distribution.

## E. Quota

All users share your project's quota: **100 writes/min** for the whole project (plus 20 uploads/min
per user). That's fine for individual researchers. For a workshop where 30 people sync at once,
stagger the syncs or request more quota in the Cloud console.
