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

# Профиль ближе к «сухой» чёткости Silero (дефолт VITS: 0.667 / 0.8 / 1.0)
_LENGTH_SCALE = float(os.environ.get("PIPER_LENGTH_SCALE", "1.15"))
_NOISE_SCALE = float(os.environ.get("PIPER_NOISE_SCALE", "0.22"))
_NOISE_W_SCALE = float(os.environ.get("PIPER_NOISE_W", "0.35"))
_CLARITY = float(os.environ.get("PIPER_CLARITY", "1.0"))  # 0=выкл, 1=как Silero

# Р/Л — сонанты, VITS их часто «съедает»
_LIQUIDS = frozenset({"r", "ɾ", "ɹ", "ʀ", "ʁ", "l", "ɭ", "ɫ", "ʎ", "ɺ", "ɻ"})
_LIQUID_MAP = {
    "ɭ": "l",  # ретрофлексный L у espeak звучит мутно в Piper
    "ɫ": "l",
    "ʎ": "l",
    "ɹ": "r",
    "ɾ": "r",
}


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


def _syn_config():
    from piper.config import SynthesisConfig

    return SynthesisConfig(
        length_scale=_LENGTH_SCALE,
        noise_scale=_NOISE_SCALE,
        noise_w_scale=_NOISE_W_SCALE,
        normalize_audio=True,
        volume=1.0,
    )


def _filter_phonemes(phonemes):
    return [p for p in phonemes if p and p not in _SKIP_PHONEMES]


def _reinforce_liquids(phonemes):
    """Делает Р/Л заметнее: нормализует аллофоны и слегка удлиняет."""
    out = []
    for i, p in enumerate(phonemes):
        p = _LIQUID_MAP.get(p, p)
        out.append(p)
        if p in ("r", "l") or p in _LIQUIDS:
            nxt = phonemes[i + 1] if i + 1 < len(phonemes) else None
            if nxt != "ː":
                out.append("ː")
    return out


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


def _boost_liquid_formants(audio, sr):
    """Лёгкий подъём 600–2000 Гц — форманты Р/Л."""
    x = _ensure_mono(audio)
    n = len(x)
    if n < 32:
        return x
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    gain = np.ones_like(freqs, dtype=np.float64)
    # плавный колокол ~1.1 кГц
    center, width = 1100.0, 900.0
    band = np.exp(-0.5 * ((freqs - center) / width) ** 2)
    gain += 0.55 * band
    y = np.fft.irfft(spec * gain, n=n).astype(np.float32)
    peak = float(np.max(np.abs(y))) + 1e-8
    y *= (float(np.max(np.abs(x))) + 1e-8) / peak
    return np.ascontiguousarray(y)


def _clarify_like_silero(audio, sr):
    """
    Постобработка под более «сухой» Silero-подобный звук:
    частично pre-emphasis + акцент формант Р/Л + нормализация.
    """
    if _CLARITY <= 0 or audio.size < 16:
        return audio
    x = _ensure_mono(audio).astype(np.float32, copy=True)
    x -= float(np.mean(x))

    coef = 0.9
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - coef * x[:-1]

    # Микс сухого VITS и подчёркнутого — ближе к Silero, без металлического ВЧ
    mix = 0.40 * min(1.0, _CLARITY)
    out = (1.0 - mix) * x + mix * y
    out = _boost_liquid_formants(out, sr)

    peak = float(np.max(np.abs(out))) + 1e-8
    out *= 0.92 / peak
    return np.ascontiguousarray(out, dtype=np.float32)


def _is_sign_letter_name(phonemes):
    """Одиночные ь/ъ espeak произносит как «мягкий/твёрдый знак»."""
    if len(phonemes) > 24:
        # длинная фраза — не режем целиком из-за ложного совпадения
        return False
    joined = "".join(phonemes)
    if "znˈɑk" in joined or "znɑk" in joined:
        if "mʲˈɑxk" in joined or "tvʲˈɵrd" in joined or "tvʲˈord" in joined:
            return True
        if "mʲɑxk" in joined:
            return True
    if phonemes[:3] == ["m", "ʲ", "ˈ"] and "z" in phonemes and "n" in phonemes:
        return True
    if phonemes[:2] == ["t", "v"] and "z" in phonemes and "n" in phonemes:
        return True
    return False


def _audio_from_phonemes(voice, phonemes, syn_config, sr):
    phonemes = _reinforce_liquids(_filter_phonemes(phonemes))
    if not phonemes or _is_sign_letter_name(phonemes):
        return None
    ids = voice.phonemes_to_ids(phonemes)
    audio = voice.phoneme_ids_to_audio(ids, syn_config=syn_config)
    if isinstance(audio, tuple):
        audio = audio[0]
    audio = _ensure_mono(audio)
    if audio.size == 0:
        return None
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.5:
        audio = audio / 32768.0
    return audio


def _synthesize_from_phonemes(voice, text):
    """
    Один непрерывный пассаж (без нарезки по словам — иначе «заикание»).
    Пробелы/пунктуация из фонем убраны; clarify один раз на весь клип.
    """
    sr = int(getattr(getattr(voice, "config", None), "sample_rate", None) or 22050)
    syn = _syn_config()

    all_phonemes = []
    for sentence in voice.phonemize(text) or []:
        filtered = _reinforce_liquids(_filter_phonemes(sentence))
        if not filtered or _is_sign_letter_name(filtered):
            continue
        all_phonemes.extend(filtered)

    if not all_phonemes:
        return np.zeros(sr // 10, dtype=np.float32), sr

    # уже усилены liquids выше — не применять reinforce дважды
    ids = voice.phonemes_to_ids(all_phonemes)
    audio = voice.phoneme_ids_to_audio(ids, syn_config=syn)
    if isinstance(audio, tuple):
        audio = audio[0]
    audio = _ensure_mono(audio)
    if audio.size == 0:
        return np.zeros(sr // 10, dtype=np.float32), sr
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.5:
        audio = audio / 32768.0
    return _clarify_like_silero(audio, sr), sr


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
    syn = _syn_config()
    if hasattr(voice, "synthesize"):
        for audio_chunk in voice.synthesize(text, syn_config=syn):
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
