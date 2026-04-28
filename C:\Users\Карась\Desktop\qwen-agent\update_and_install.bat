@echo off
chcp 65001 >nul
echo ==========================================
echo Обновление проекта qwen-agent через Git
echo ==========================================
echo.

cd /d C:\Users\Карась\Desktop\qwen-agent

echo [1/4] Очистка папки от старых файлов...
del /s /q *.* 2>nul
for /d %%i in (*) do rmdir /s /q "%%i" 2>nul
timeout /t 2 /nobreak >nul

echo [2/4] Клонирование репозитория...
git clone https://github.com/QwenLM/qwen-agent.git temp_repo
if %errorlevel% neq 0 (
    echo ОШИБКА: Не удалось клонировать репозиторий. Проверьте подключение к интернету и установлен ли Git.
    pause
    exit /b 1
)

echo [3/4] Перемещение файлов в текущую папку...
xcopy /E /I /Y temp_repo\* . 
if %errorlevel% neq 0 (
    echo ОШИБКА: Не удалось переместить файлы.
    pause
    exit /b 1
)

rmdir /s /q temp_repo

echo [4/4] Установка зависимостей...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ПРЕДУПРЕЖДЕНИЕ: Возможны проблемы с установкой некоторых пакетов.
)

echo.
echo ==========================================
echo Готово! Проект обновлен.
echo Для запуска введите: python run_agent.py --mode autonomous
echo ==========================================
pause
