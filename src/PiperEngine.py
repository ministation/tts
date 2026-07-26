"""
Piper ONNX runtime — быстрый синтез на CPU.
"""
import io
import os
import threading
from functools import lru_cache

import numpy as np

_lock = threading.Lock()
_voices = {}


def is_piper_available():
    try:
        import piper  # noqa: F401
        return True
    except ImportError:
        return False


def get_model_paths(speaker_id, voices_dir=None):
    voices_dir = voices_dir or os.path.join(os.getcwd(), "voices")
    onnx = os.path.join(voices_dir, f"{speaker_id}.onnx")
    # piper expects either name.onnx.json or name.json next to onnx
    for suffix in (".onnx.json", ".json"):
        cfg = os.path.join(voices_dir, f"{speaker_id}{suffix}")
        if os.path.isfile(cfg):
            return onnx, cfg
    return onnx, os.path.join(voices_dir, f"{speaker_id}.onnx.json")


def has_piper_model(speaker_id, voices_dir=None):
    onnx, cfg = get_model_paths(speaker_id, voices_dir)
    return os.path.isfile(onnx) and (os.path.isfile(cfg) or os.path.isfile(onnx + ".json"))


def _load_voice(onnx_path):
    from piper import PiperVoice

    with _lock:
        voice = _voices.get(onnx_path)
        if voice is None:
            if not os.path.isfile(onnx_path):
                raise FileNotFoundError(f"Piper model not found: {onnx_path}")
            voice = PiperVoice.load(onnx_path)
            _voices[onnx_path] = voice
        return voice


def synthesize(text, speaker_id=None, onnx_path=None, sample_rate=22050):
    """Return float32 mono numpy audio and actual sample rate."""
    if onnx_path is None:
        if not speaker_id:
            raise ValueError("speaker_id or onnx_path required")
        onnx_path, _ = get_model_paths(speaker_id)
    voice = _load_voice(onnx_path)

    chunks = []
    sr = getattr(voice, "sample_rate", None) or sample_rate
    # piper-tts API variants
    if hasattr(voice, "synthesize"):
        for audio_chunk in voice.synthesize(text):
            if hasattr(audio_chunk, "audio_float_array"):
                chunks.append(np.asarray(audio_chunk.audio_float_array, dtype=np.float32))
                sr = getattr(audio_chunk, "sample_rate", sr)
            elif hasattr(audio_chunk, "audio"):
                raw = np.asarray(audio_chunk.audio, dtype=np.float32)
                # int16 PCM in some versions
                if raw.dtype == np.int16 or raw.max() > 1.5:
                    raw = raw.astype(np.float32) / 32768.0
                chunks.append(raw)
                sr = getattr(audio_chunk, "sample_rate", sr)
            else:
                arr = np.asarray(audio_chunk, dtype=np.float32)
                if arr.max() > 1.5:
                    arr = arr / 32768.0
                chunks.append(arr)
    elif hasattr(voice, "synthesize_stream_raw"):
        for raw in voice.synthesize_stream_raw(text):
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            chunks.append(arr)
    else:
        raise RuntimeError("Unsupported piper-tts API")

    if not chunks:
        return np.zeros(sr, dtype=np.float32), sr
    audio = np.concatenate(chunks)
    return audio, int(sr)


def synthesize_to_wav_bytes(text, speaker_id=None, onnx_path=None, target_sr=None):
    import soundfile as sf

    audio, sr = synthesize(text, speaker_id=speaker_id, onnx_path=onnx_path)
    if target_sr and target_sr != sr and len(audio) > 1:
        duration = len(audio) / sr
        new_len = max(1, int(duration * target_sr))
        audio = np.interp(
            np.linspace(0, len(audio) - 1, new_len),
            np.arange(len(audio)),
            audio,
        )
        sr = target_sr
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue(), sr
