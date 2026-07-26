"""
Piper ONNX runtime — быстрый синтез на CPU.
"""
import io
import os
import threading

import numpy as np

_lock = threading.Lock()
_voices = {}

# Эти «фонемы» espeak/Piper озвучивают как щелчки/«пробел»/знаки — выкидываем.
_SKIP_PHONEMES = frozenset(
    {
        " ",
        "_",
        "^",
        "$",
        "!",
        "'",
        "(",
        ")",
        ",",
        "-",
        ".",
        ":",
        ";",
        "?",
        "#",
        '"',
        "↓",
        "\t",
        "\n",
        "\r",
    }
)


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


def _filter_phonemes(phonemes):
    return [p for p in phonemes if p and p not in _SKIP_PHONEMES]


def _ensure_mono(audio):
    """Гарантированно mono float32 shape (n,). Иначе sf.write(1, n) → N каналов."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 0:
        return np.zeros(1, dtype=np.float32)
    if audio.ndim == 1:
        return np.ascontiguousarray(audio)
    # (channels, samples) или (samples, channels)
    if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[-1]:
        audio = audio.mean(axis=0)
    else:
        audio = audio.mean(axis=-1)
    return np.ascontiguousarray(audio.reshape(-1))


def _synthesize_from_phonemes(voice, text):
    """Синтез без фонем пробела/пунктуации (иначе слышны «стыки»)."""
    chunks = []
    sr = int(getattr(getattr(voice, "config", None), "sample_rate", None) or 22050)
    sentences = voice.phonemize(text) or []
    for sentence in sentences:
        phonemes = _filter_phonemes(sentence)
        if not phonemes:
            continue
        ids = voice.phonemes_to_ids(phonemes)
        audio = _ensure_mono(voice.phoneme_ids_to_audio(ids))
        if audio.size == 0:
            continue
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.5:
            audio = audio / 32768.0
        chunks.append(audio)
    if not chunks:
        return np.zeros(sr // 10, dtype=np.float32), sr
    return _ensure_mono(np.concatenate(chunks)), sr


def synthesize(text, speaker_id=None, onnx_path=None, sample_rate=22050):
    """Return float32 mono numpy audio and actual sample rate."""
    if onnx_path is None:
        if not speaker_id:
            raise ValueError("speaker_id or onnx_path required")
        onnx_path, _ = get_model_paths(speaker_id)
    voice = _load_voice(onnx_path)

    text = (text or "").strip()
    if not text:
        sr = int(getattr(getattr(voice, "config", None), "sample_rate", None) or sample_rate)
        return np.zeros(sr // 10, dtype=np.float32), sr

    # Предпочитаем путь через фонемы без пробелов/знаков
    if hasattr(voice, "phonemize") and hasattr(voice, "phonemes_to_ids"):
        return _synthesize_from_phonemes(voice, text)

    chunks = []
    sr = getattr(voice, "sample_rate", None) or sample_rate
    if hasattr(voice, "synthesize"):
        for audio_chunk in voice.synthesize(text):
            if hasattr(audio_chunk, "audio_float_array"):
                chunks.append(_ensure_mono(audio_chunk.audio_float_array))
                sr = getattr(audio_chunk, "sample_rate", sr)
            elif hasattr(audio_chunk, "audio"):
                raw = _ensure_mono(audio_chunk.audio)
                peak = float(np.max(np.abs(raw))) if raw.size else 0.0
                if peak > 1.5:
                    raw = raw / 32768.0
                chunks.append(raw)
                sr = getattr(audio_chunk, "sample_rate", sr)
            else:
                arr = _ensure_mono(audio_chunk)
                peak = float(np.max(np.abs(arr))) if arr.size else 0.0
                if peak > 1.5:
                    arr = arr / 32768.0
                chunks.append(arr)
    elif hasattr(voice, "synthesize_stream_raw"):
        for raw in voice.synthesize_stream_raw(text):
            arr = _ensure_mono(np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)
            chunks.append(arr)
    else:
        raise RuntimeError("Unsupported piper-tts API")

    if not chunks:
        return np.zeros(int(sr), dtype=np.float32), int(sr)
    return _ensure_mono(np.concatenate(chunks)), int(sr)


def synthesize_to_wav_bytes(text, speaker_id=None, onnx_path=None, target_sr=None):
    import soundfile as sf

    audio, sr = synthesize(text, speaker_id=speaker_id, onnx_path=onnx_path)
    audio = _ensure_mono(audio)
    if target_sr and target_sr != sr and len(audio) > 1:
        duration = len(audio) / sr
        new_len = max(1, int(duration * target_sr))
        audio = np.interp(
            np.linspace(0, len(audio) - 1, new_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
        sr = target_sr
    buf = io.BytesIO()
    sf.write(buf, _ensure_mono(audio), sr, format="WAV")
    return buf.getvalue(), sr
