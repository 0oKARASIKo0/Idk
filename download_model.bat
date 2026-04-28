@echo off
chcp 65001 >nul
:: Download Local LLM Model from HuggingFace

cd /d "%~dp0"

echo ============================================
echo   Download Local LLM Model
echo ============================================
echo.

echo Available models:
echo   1. Qwen 3 0.6B   - 1.3GB - Fast, Russian
echo   2. Qwen 3 3B     - 6.0GB - Good balance, Russian
echo   3. Qwen 3.5 3B   - 6.5GB - NEWEST, best Russian (RECOMMENDED)
echo   4. Qwen 3.5 7B   - 14GB  - Maximum quality, needs GPU
echo   5. Qwen 2.5 3B   - 6.0GB - Stable, Russian
echo.

set /p choice="Choose model (1-5, default=3): "

if "%choice%"=="" set choice=3

if "%choice%"=="1" set model=qwen3-0.6b
if "%choice%"=="2" set model=qwen3-3b
if "%choice%"=="3" set model=qwen3.5-3b
if "%choice%"=="4" set model=qwen3.5-7b
if "%choice%"=="5" set model=qwen2.5-3b

echo.
echo Downloading %model%...
echo This may take several minutes...
echo.

python download_model.py --model %model%

echo.
pause
