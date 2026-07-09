"""
XTTS v2 worker — клонирование голоса по reference.wav.
Запуск: clone_venv/Scripts/python.exe clone_worker.py
"""
import base64
import datetime
import importlib.metadata
import io
import os
import tempfile
import traceback

import soundfile as sf
import torch
import torchaudio
from flask import Flask, jsonify, request, abort

app = Flask(__name__)

PORT = int(os.environ.get("CLONE_PORT", "5001"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLONE_SPEED = float(os.environ.get("CLONE_SPEED", "1.8"))
_model = None
_audio_patch_applied = False
os.environ.setdefault("COQUI_TOS_AGREED", "1")


def _patch_xtts_audio_loading():
    """torchaudio 2.9+ loads files via torchcodec, which needs FFmpeg DLLs on Windows."""
    global _audio_patch_applied
    if _audio_patch_applied:
        return

    from TTS.tts.models import xtts as xtts_module

    def load_audio(audiopath, sampling_rate):
        audio, lsr = sf.read(audiopath, dtype="float32", always_2d=True)
        audio = torch.from_numpy(audio.T)
        if audio.size(0) != 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if lsr != sampling_rate:
            audio = torchaudio.functional.resample(audio, lsr, sampling_rate)
        audio.clip_(-1, 1)
        return audio

    xtts_module.load_audio = load_audio
    _audio_patch_applied = True


def get_model():
    global _model
    if _model is None:
        _patch_xtts_audio_loading()
        from TTS.api import TTS

        print(f"Loading XTTS v2 on {DEVICE} (first run downloads ~2 GB)...", flush=True)
        _model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
        print("XTTS ready", flush=True)
    return _model


def _tts_model():
    return get_model().synthesizer.tts_model


def _load_pt(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def _voice_model_path(reference=None, speaker_id=None, voice_path=None):
    if voice_path:
        return os.path.abspath(voice_path)
    if speaker_id:
        return os.path.join(os.getcwd(), "voices", f"{speaker_id}.pt")
    if reference:
        speaker_id = os.path.basename(os.path.dirname(os.path.abspath(reference)))
        return os.path.join(os.getcwd(), "voices", f"{speaker_id}.pt")
    return None


def _normalize_references(reference):
    if reference is None:
        return []
    if isinstance(reference, str):
        refs = [reference]
    elif isinstance(reference, list):
        refs = reference
    else:
        refs = [reference]
    return [os.path.abspath(r) for r in refs if isinstance(r, str) and r and os.path.isfile(r)]


def save_voice_model(reference, voice_path=None, speaker_id=None):
    refs = _normalize_references(reference)
    if not refs:
        raise ValueError("reference required")
    voice_path = _voice_model_path(reference=refs[0], speaker_id=speaker_id, voice_path=voice_path)
    if not voice_path:
        raise ValueError("voice_path or reference required")
    speaker_id = speaker_id or os.path.basename(os.path.dirname(refs[0]))
    tts_model = _tts_model()
    speaker_wav = refs if len(refs) > 1 else refs[0]
    voice, model_metadata = tts_model._clone_voice(speaker_wav)
    from TTS.utils.voices import VoiceMetadata

    metadata = VoiceMetadata(
        model=model_metadata,
        speaker_id=speaker_id,
        source_files=refs,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="minutes"),
        coqui_version=importlib.metadata.version("coqui-tts"),
    )
    voice_dict = {**voice, "metadata": metadata.to_dict(), "engine": "xtts"}
    os.makedirs(os.path.dirname(voice_path) or ".", exist_ok=True)
    torch.save(voice_dict, voice_path)
    return voice_path


def _synthesize_to_file(text, language, reference=None, voice_path=None, speaker_id=None):
    out_path = tempfile.mktemp(suffix=".wav")
    refs = _normalize_references(reference)
    voice_path = _voice_model_path(
        reference=refs[0] if refs else None,
        speaker_id=speaker_id,
        voice_path=voice_path,
    )
    if voice_path and os.path.isfile(voice_path):
        voice = _load_pt(voice_path)
        result = _tts_model().inference(
            text,
            language,
            voice["gpt_conditioning_latents"],
            voice["speaker_embedding"],
            speed=CLONE_SPEED,
        )
        sf.write(out_path, result["wav"], 24000)
        return out_path

    if not refs:
        raise FileNotFoundError(f"Voice model not found: {voice_path}")

    speaker_wav = refs if len(refs) > 1 else refs[0]
    get_model().tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language=language,
        file_path=out_path,
        speed=CLONE_SPEED,
    )
    return out_path


@app.route("/health")
def health():
    return jsonify({"ok": True, "device": DEVICE, "model_loaded": _model is not None})


@app.route("/warmup", methods=["POST"])
def warmup():
    get_model()
    return jsonify({"ok": True, "model_loaded": True})


@app.route("/encode", methods=["POST"])
def encode():
    data = request.json or {}
    references = data.get("references") or data.get("reference")
    voice_path = data.get("voice_model")
    speaker_id = data.get("speaker_id")

    refs = _normalize_references(references)
    if not refs:
        abort(400, description="Missing or invalid reference wav path(s)")

    try:
        saved = save_voice_model(refs, voice_path, speaker_id=speaker_id)
        return jsonify({"ok": True, "voice_model": saved, "references": len(refs)})
    except Exception as exc:
        print(f"XTTS encode error: {exc}", flush=True)
        traceback.print_exc()
        return jsonify({"ok": False, "description": str(exc)}), 500


@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.json or {}
    text = data.get("text")
    references = data.get("references") or data.get("reference")
    voice_path = data.get("voice_model")
    speaker_id = data.get("speaker_id")
    language = data.get("language", "ru")

    if not text:
        abort(400, description="Missing text")

    refs = _normalize_references(references)
    resolved_voice = _voice_model_path(
        reference=refs[0] if refs else None,
        speaker_id=speaker_id,
        voice_path=voice_path,
    )
    has_model = resolved_voice and os.path.isfile(resolved_voice)
    if not refs and not has_model:
        abort(400, description="Missing voice model or reference wav path(s)")

    out_path = None
    try:
        out_path = _synthesize_to_file(
            text,
            language,
            reference=refs if refs else None,
            voice_path=voice_path,
            speaker_id=speaker_id,
        )
        audio, sample_rate = sf.read(out_path)
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV")
        return jsonify(
            {
                "audio": base64.b64encode(buf.getvalue()).decode(),
                "sample_rate": int(sample_rate),
            }
        )
    except Exception as exc:
        print(f"XTTS synthesize error: {exc}", flush=True)
        traceback.print_exc()
        return jsonify({"ok": False, "description": str(exc)}), 500
    finally:
        if out_path and os.path.isfile(out_path):
            os.remove(out_path)


if __name__ == "__main__":
    print(f"XTTS worker starting on http://127.0.0.1:{PORT}", flush=True)
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
