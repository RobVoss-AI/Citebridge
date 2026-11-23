@echo off
REM ============================================
REM CiteBridge v2 — Setup (Windows)
REM Zotero ^<-^> Gemini Notebook (formerly NotebookLM)
REM ============================================
setlocal

echo.
echo ========================================
echo   CiteBridge Setup
echo   Zotero ^<-^> Gemini Notebook
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.10+ is required. Install it from https://www.python.org/downloads/
    pause
    exit /b 1
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.10 or newer is required.
    pause
    exit /b 1
)

echo Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q
echo Dependencies installed.

if "%CITEBRIDGE_HOME%"=="" set "CITEBRIDGE_HOME=%USERPROFILE%\.citebridge"
if not exist "%CITEBRIDGE_HOME%" mkdir "%CITEBRIDGE_HOME%"
if exist "%CITEBRIDGE_HOME%\oauth_client.json" (
    echo Google sign-in file already in place.
) else if exist "oauth_client.json" (
    copy /Y "oauth_client.json" "%CITEBRIDGE_HOME%\oauth_client.json" >nul
    echo Google sign-in file installed.
) else (
    echo WARNING: No oauth_client.json found. Copy the one from your CiteBridge download to:
    echo     %CITEBRIDGE_HOME%\oauth_client.json
)

echo.
set /p EXP="Install the EXPERIMENTAL notes/text read-back add-on? [y/N] "
if /I "%EXP%"=="y" (
    pip install -r requirements-experimental.txt -q
    playwright install chromium
    echo A browser will open. Sign in to Google, wait for Gemini Notebook to load,
    echo then come back here and press Enter.
    notebooklm login
)

echo.
echo ========================================
echo   Setup complete!
echo ========================================
echo Start CiteBridge any time with:
echo     .venv\Scripts\activate
echo     streamlit run app.py
echo Then click "Sign in with Google" and pick your Zotero collections.
echo.
pause
