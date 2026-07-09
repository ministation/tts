# Установка XTTS (клонирование голоса) — Python 3.11+
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $Root

$py311 = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
if (-not (Test-Path $py311)) {
    Write-Host "Python 3.11 не найден. Установите: winget install Python.Python.3.11"
    exit 1
}

Write-Host "Создаю clone_venv..."
if (Test-Path clone_venv) { Remove-Item -Recurse -Force clone_venv }
& $py311 -m venv clone_venv

$pip = Join-Path $Root "clone_venv\Scripts\pip.exe"
& $pip install torch torchaudio "transformers>=4.46,<5" "coqui-tts[codec]" flask requests

Write-Host ""
Write-Host "Готово! XTTS в clone_venv/"
Write-Host "Worker: clone_venv\Scripts\python.exe clone_worker.py"
