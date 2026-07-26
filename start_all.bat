@echo off
cd /d %~dp0
start "SS14 TTS API" cmd /k python ss14tts.py
echo.
echo Запущен SS14 TTS API: http://127.0.0.1:8000
echo Обучение голосов (Piper): http://127.0.0.1:8000/train
echo.
echo XTTS worker больше не нужен для новых голосов.
echo Если нужен legacy XTTS: start_clone_worker.bat
echo.
pause
