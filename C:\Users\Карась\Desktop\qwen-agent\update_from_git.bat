@echo off
chcp 65001 >nul
echo ========================================
echo Очистка и обновление проекта из Git
echo ========================================
echo.

cd /d C:\Users\Карась\Desktop\qwen-agent

echo [1/3] Удаление всех файлов и папок (кроме этого скрипта)...
for /f "delims=" %%i in ('dir /b /a-d') do (
    if not "%%i"=="%~nx0" del /q "%%i"
)
for /f "delims=" %%i in ('dir /b /ad') do (
    if not "%%i"=="%~nx0" rmdir /s /q "%%i"
)

echo [2/3] Клонирование репозитория...
REM Замените URL на актуальный адрес вашего репозитория, если он другой
git clone https://github.com/QwenLM/qwen-agent.git temp_clone

echo [3/3] Перемещение файлов в текущую папку...
move temp_clone\* .
move temp_clone\.* . 2>nul
rmdir /s /q temp_clone

echo.
echo ========================================
echo Готово! Проект обновлен.
echo Теперь установите зависимости:
echo pip install -r requirements.txt
echo ========================================
pause
