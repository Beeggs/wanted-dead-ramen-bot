@echo off
chcp 65001 >nul
cd /d "%~dp0"

set IP=%~1
if "%IP%"=="" (
    set /p IP=IP-адрес консоли (например 192.168.1.50): 
)

if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe ramen_bot.py --run %IP% --trophy
) else (
    python ramen_bot.py --run %IP% --trophy
)

pause
