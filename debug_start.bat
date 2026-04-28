@echo off
:: Qwen AI Agent - Debug launcher

cd /d "%~dp0"

echo ============================================
echo   Qwen AI Agent - DEBUG MODE
echo ============================================
echo.
echo This will show detailed output...
echo.
echo [1/3] Python check...
python --version
echo.

echo [2/3] Module check...
python -c "import qwen_agent; print('OK: qwen_agent')" 2>&1
python -c "import openai; print('OK: openai')" 2>&1
python -c "import torch; print('OK: torch')" 2>&1
echo.

echo [3/3] Starting agent with FULL OUTPUT...
echo If you see errors below, they are the problem.
echo.
echo ============================================
echo.

:: Run with full verbose output and redirect stderr to file
python -u run_agent.py --mode interactive >agent_output.log 2>&1

echo.
echo ============================================
echo   Agent exited. Check agent_output.log
echo ============================================
echo.
echo Last 30 lines of output:
echo.
powershell -Command "Get-Content agent_output.log -Tail 30"
echo.
pause
