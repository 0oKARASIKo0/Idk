@echo off
:: Qwen AI Agent - Install and Launch

cd /d "%~dp0"

echo ============================================
echo   Qwen AI Agent - Install and Launch
echo ============================================
echo.

:: Check Python
echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    pause
    exit /b 1
)
echo [OK] Python found
echo.

:: Install dependencies
echo [2/5] Installing dependencies...
pip install -r requirements.txt >install.log 2>&1
if errorlevel 1 (
    echo [!] Install failed, check install.log
) else (
    echo [OK] Dependencies ready
)
echo.

:: Check Ollama
echo [3/5] Checking Ollama...
where ollama >nul 2>&1
if errorlevel 1 (
    echo [!] Ollama not found - agent will have limited function
) else (
    echo [OK] Ollama found
)
echo.

:: Launch agent with output visible
echo [4/5] Starting agent...
echo.
echo ============================================
echo   AGENT IS RUNNING
echo   DO NOT CLOSE THIS WINDOW
echo   Press Ctrl+C to stop
echo ============================================
echo.

python run_agent.py --mode interactive

echo.
echo ============================================
echo   Agent finished
echo ============================================
echo.
echo Press ENTER to close this window...
pause >nul

