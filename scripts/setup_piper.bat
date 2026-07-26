@echo off
cd /d %~dp0\..
echo Installing Piper / TTS dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if not exist piper_venv (
  python -m venv piper_venv
)
piper_venv\Scripts\python.exe -m pip install --upgrade pip
piper_venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
piper_venv\Scripts\python.exe -m pip install piper-tts onnxruntime huggingface_hub faster-whisper soundfile numpy PyYAML lightning tensorboard librosa onnx
echo.
echo Done. Run: python ss14tts.py
echo Train UI: http://127.0.0.1:5000/train
echo If espeak missing: winget install eSpeak-NG.eSpeak-NG
pause
