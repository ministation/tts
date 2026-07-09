"""
XTTS v2 worker — клонирование голоса по reference.wav.
Запуск: clone_venv/Scripts/python.exe clone_worker.py
"""
import base64
import io
import os
import tempfile

from flask import Flask, jsonify, request, abort

app = Flask(__name__)

PORT = int(os.environ.get("CLONE_PORT", "5001"))
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
_model = None


def get_model():
    global _model
    if _model is None:
        from TTS.api import TTS

        print(f"Loading XTTS v2 on {DEVICE} (first run downloads ~2 GB)...", flush=True)
        _model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
        print("XTTS ready", flush=True)
    return _model


@app.route("/health")
def health():
    return jsonify({"ok": True, "device": DEVICE, "model_loaded": _model is not None})


@app.route("/warmup", methods=["POST"])
def warmup():
    get_model()
    return jsonify({"ok": True, "model_loaded": True})


@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.json or {}
    text = data.get("text")
    reference = data.get("reference")
    language = data.get("language", "ru")

    if not text:
        abort(400, description="Missing text")
    if not reference or not os.path.isfile(reference):
        abort(400, description="Missing or invalid reference wav path")

    out_path = tempfile.mktemp(suffix=".wav")
    try:
        get_model().tts_to_file(
            text=text,
            speaker_wav=reference,
            language=language,
            file_path=out_path,
        )
        import soundfile as sf

        audio, sample_rate = sf.read(out_path)
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV")
        return jsonify(
            {
                "audio": base64.b64encode(buf.getvalue()).decode(),
                "sample_rate": int(sample_rate),
            }
        )
    finally:
        if os.path.isfile(out_path):
            os.remove(out_path)


if __name__ == "__main__":
    print(f"XTTS worker starting on http://127.0.0.1:{PORT}", flush=True)
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
