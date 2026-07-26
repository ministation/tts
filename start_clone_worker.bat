@echo off
cd /d %~dp0
echo [LEGACY] XTTS worker — для новых голосов используйте Piper (/train).
echo Starting XTTS worker on http://127.0.0.1:5001
clone_venv\Scripts\python.exe clone_worker.py
pause
