#!/usr/bin/env python3
"""
Phase 0 smoke test for the official Gemini Notebook API.

Signs in (or reuses a saved sign-in), then in ONE reusable notebook named
"CiteBridge smoke test" — the API can't delete notebooks, so we never make
more than one — it:
  1. uploads a 1-page PDF with a custom display name (resumable upload),
  2. adds a Markdown text source,
  3. polls both until ACTIVE/FAILED,
and prints the raw JSON the API returned so field names can be checked.

Usage:
    python scripts/smoke_official.py              # sign in if needed
    python scripts/smoke_official.py --account you@gmail.com
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from citebridge.notebook import auth  # noqa: E402
from citebridge.notebook.official import OfficialNotebookClient, parse_source  # noqa: E402

NOTEBOOK_NAME = "CiteBridge smoke test"


def tiny_pdf(text: str) -> bytes:
    """A valid one-page PDF built by hand (no dependencies)."""
    stream = f"BT /F1 18 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", help="Google account already signed in to CiteBridge")
    args = ap.parse_args()

    accounts = auth.signed_in_accounts()
    account = args.account or (accounts[0] if accounts else None)
    if not account:
        print("Opening your browser to sign in to Google…")
        account = auth.sign_in()
    print(f"✓ Signed in as {account}")

    client = OfficialNotebookClient(auth.token_provider(account))
    nb = next((n for n in client.list_notebooks() if n.display_name == NOTEBOOK_NAME), None)
    if nb is None:
        nb = client.create_notebook(NOTEBOOK_NAME)
        print(f"✓ Created notebook {nb.name}")
    else:
        print(f"✓ Reusing notebook {nb.name}")
    print(f"  {nb.url}")

    stamp = time.strftime("%Y-%m-%d %H:%M")
    with tempfile.TemporaryDirectory() as d:
        pdf = Path(d) / "smoke.pdf"
        pdf.write_bytes(tiny_pdf(f"CiteBridge smoke test {stamp}"))
        up = client.upload_file(nb.name, pdf, f"Smoke (2026) — PDF upload test {stamp}")
    print(f"✓ Uploaded PDF → {up.name} [{up.state.value}] display name: {up.display_name!r}")

    md = client.add_text(nb.name, f"Smoke (2026) — Markdown test {stamp}",
                         f"# Smoke test\n\nAdded by CiteBridge at {stamp}.")
    print(f"✓ Added Markdown → {md.name} [{md.state.value}]")

    print("… waiting for processing")
    done = client.wait_for([up.name, md.name], timeout=300)
    ok = True
    for name in (up.name, md.name):
        src = done.get(name)
        if src is None:
            print(f"✗ {name} still processing after 5 minutes")
            ok = False
            continue
        print(f"{'✓' if src.state.value == 'ACTIVE' else '✗'} {src.display_name}: {src.state.value}"
              + (f" — {src.failure_reason}" if src.state.value == "FAILED" else ""))
        ok &= src.state.value == "ACTIVE"

    raw = client._call("GET", f"/v1/{up.name}")
    print("\nRaw GetSource JSON (check field names):")
    print(json.dumps(raw, indent=2))
    assert parse_source(raw).name == up.name
    print("\n" + ("✅ Smoke test passed." if ok else "⚠️  Smoke test finished with problems."))
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except auth.AuthError as e:
        print(f"✗ {e}")
        sys.exit(2)
