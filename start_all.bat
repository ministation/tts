@echo off
cd /d %~dp0
start "SS14 TTS API" cmd /k python ss14tts.py
timeout /t 3 /nobreak >nul
start "XTTS Worker" cmd /k clone_venv\Scripts\python.exe clone_worker.py
echo.
echo Запущено два окна:
echo   1. SS14 TTS API  - http://127.0.0.1:5000
echo   2. XTTS Worker   - http://127.0.0.1:5001
echo.
echo Для клонирования голосов: http://127.0.0.1:5000/train
pause
