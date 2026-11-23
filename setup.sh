#!/bin/bash
# ============================================
# CiteBridge v2 — Setup (macOS / Linux)
# Zotero <-> Gemini Notebook (formerly NotebookLM)
# Usage:  chmod +x setup.sh && ./setup.sh
# ============================================
set -e

echo ""
echo "========================================"
echo "  CiteBridge Setup"
echo "  Zotero <-> Gemini Notebook"
echo "========================================"
echo ""

# Python 3.10+
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3.10+ is required. Install it from https://www.python.org/downloads/"
    exit 1
fi
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "ERROR: Python 3.10 or newer is required (found $(python3 --version))."
    exit 1
fi
echo "✅ Found $(python3 --version)"

echo ""
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✅ Dependencies installed"

# Google sign-in file (ships with the CiteBridge download, never in git)
CB_HOME="${CITEBRIDGE_HOME:-$HOME/.citebridge}"
mkdir -p "$CB_HOME"
if [ -f "oauth_client.json" ] && [ ! -f "$CB_HOME/oauth_client.json" ]; then
    cp oauth_client.json "$CB_HOME/oauth_client.json"
    chmod 600 "$CB_HOME/oauth_client.json"
    echo "✅ Google sign-in file installed"
elif [ -f "$CB_HOME/oauth_client.json" ]; then
    echo "✅ Google sign-in file already in place"
else
    echo "⚠️  No oauth_client.json found. Copy the one from your CiteBridge download to:"
    echo "     $CB_HOME/oauth_client.json"
fi

echo ""
read -p "Install the EXPERIMENTAL notes/text read-back add-on? [y/N] " exp
if [[ "$exp" =~ ^[Yy]$ ]]; then
    pip install -r requirements-experimental.txt -q
    playwright install chromium
    echo ""
    echo "A browser will open — sign in to Google, wait for Gemini Notebook to load,"
    echo "then come back here and press Enter."
    notebooklm login
    echo "✅ Experimental add-on ready (its login expires every few weeks)"
fi

echo ""
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo ""
echo "Start CiteBridge any time with:"
echo "    source .venv/bin/activate"
echo "    streamlit run app.py"
echo ""
echo "Then click 'Sign in with Google' and pick your Zotero collections."
echo "Tip: keep Zotero 7 desktop open — CiteBridge reads your library from it directly."
echo ""
