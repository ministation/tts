# Установка Piper TTS (быстрый CPU-синтез) + Whisper для обучения
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $Root

Write-Host "Устанавливаю Piper runtime (piper-tts, onnxruntime)..."
python -m pip install --upgrade pip
python -m pip install "piper-tts" onnxruntime huggingface_hub faster-whisper soundfile numpy

$py311 = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
if (-not (Test-Path $py311)) {
    $py311 = (Get-Command python -ErrorAction SilentlyContinue).Source
}

Write-Host "Создаю piper_venv для обучения (может занять время)..."
if (-not (Test-Path piper_venv)) {
    & $py311 -m venv piper_venv
}

$pip = Join-Path $Root "piper_venv\Scripts\pip.exe"
$py = Join-Path $Root "piper_venv\Scripts\python.exe"

& $pip install --upgrade pip
& $pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
& $pip install "piper-tts" onnxruntime huggingface_hub faster-whisper soundfile numpy PyYAML
# Training stack (Lightning VITS). Если упадёт — runtime всё равно работает.
try {
    & $pip install "lightning" "tensorboard" "librosa" "onnx" "onnxruntime"
    Write-Host "Пробую установить piper training из OHF-Voice/piper1-gpl..."
    & $pip install "piper-train @ git+https://github.com/OHF-Voice/piper1-gpl.git"
} catch {
    Write-Host "Предупреждение: полный piper-train не установился. Обучение попробует альтернативный путь."
}

Write-Host ""
Write-Host "Проверка espeak-ng (нужен для фонемы)..."
$espeak = Get-Command espeak-ng -ErrorAction SilentlyContinue
if (-not $espeak) {
    Write-Host "espeak-ng не найден. Установите: winget install eSpeak-NG.eSpeak-NG"
    Write-Host "Или скачайте с https://github.com/espeak-ng/espeak-ng/releases"
} else {
    Write-Host "espeak-ng найден: $($espeak.Source)"
}

Write-Host ""
Write-Host "Готово!"
Write-Host "  Runtime: python ss14tts.py"
Write-Host "  Train venv: piper_venv\Scripts\python.exe"
Write-Host "  Обучение голосов: http://127.0.0.1:5000/train"
