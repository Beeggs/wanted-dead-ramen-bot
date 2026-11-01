@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo === Установка окружения бота ===
where python >nul 2>nul
if errorlevel 1 (
    echo Python не найден. Поставь Python 3.10 или новее с python.org
    echo и при установке отметь галочку "Add Python to PATH".
    pause
    exit /b 1
)

if not exist .venv (
    echo Создаю виртуальное окружение .venv ...
    python -m venv .venv
)

echo Обновляю pip и ставлю зависимости ...
call .venv\Scripts\python.exe -m pip install --upgrade pip
call .venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Установка не удалась. Проверь интернет и версию Python.
    pause
    exit /b 1
)

echo.
echo Готово. Дальше:
echo   1) .venv\Scripts\python.exe ramen_bot.py --discover
echo   2) .venv\Scripts\python.exe ramen_bot.py --register IP_КОНСОЛИ
echo   3) run.bat
pause
