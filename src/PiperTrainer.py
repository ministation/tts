"""
Офлайн-обучение Piper: аудио → Whisper → fine-tune → ONNX.
"""
import csv
import glob
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf

PIPER_SR = 22050
MIN_CHUNK_SEC = 1.5
MAX_CHUNK_SEC = 12.0
DEFAULT_EPOCHS = int(os.environ.get("PIPER_TRAIN_EPOCHS", "200"))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
BASE_VOICE = "ru_RU-irina-medium"
HF_VOICE_REPO = "rhasspy/piper-voices"
HF_VOICE_PATH = f"ru/ru_RU/irina/medium/{BASE_VOICE}"


def _ffmpeg():
    cwd = os.getcwd()
    for name in ("ffmpeg.exe", "ffmpeg"):
        local = os.path.join(cwd, "bin", name)
        if os.path.isfile(local):
            return local
    return "ffmpeg"


def _piper_python():
    local = os.path.join(os.getcwd(), "piper_venv", "Scripts", "python.exe")
    if os.path.isfile(local):
        return local
    return os.environ.get("PIPER_PYTHON", "python")


def is_piper_train_available():
    py = _piper_python()
    try:
        proc = subprocess.run(
            [py, "-c", "import piper; print('ok')"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return proc.returncode == 0
    except Exception:
        return False


def is_whisper_available():
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def piper_status():
    from src.PiperEngine import is_piper_available, has_piper_model

    return {
        "piper_runtime": is_piper_available(),
        "whisper": is_whisper_available(),
        "train_venv": os.path.isfile(
            os.path.join(os.getcwd(), "piper_venv", "Scripts", "python.exe")
        ),
        "espeak": _espeak_ok(),
    }


def _espeak_ok():
    for name in ("espeak-ng", "espeak"):
        try:
            proc = subprocess.run(
                [name, "--version"], capture_output=True, text=True, timeout=10
            )
            if proc.returncode == 0:
                return True
        except Exception:
            pass
    return False


def _to_wav_mono(input_path, output_path, sr=PIPER_SR):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = [
        _ffmpeg(),
        "-y",
        "-i",
        input_path,
        "-ar",
        str(sr),
        "-ac",
        "1",
        "-af",
        "loudnorm",
        output_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode == 0 and os.path.isfile(output_path):
            return output_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    import librosa

    audio, _ = librosa.load(input_path, sr=sr, mono=True)
    peak = np.max(np.abs(audio)) or 1.0
    audio = (audio / peak) * 0.95
    sf.write(output_path, audio, sr)
    return output_path


def _split_wav(path, out_dir, min_sec=MIN_CHUNK_SEC, max_sec=MAX_CHUNK_SEC):
    """Нарезка по тишине / фиксированным окнам."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    os.makedirs(out_dir, exist_ok=True)

    # простая энергия для пауз
    frame = int(0.02 * sr)
    hop = frame
    energies = []
    for i in range(0, len(audio) - frame, hop):
        energies.append(float(np.sqrt(np.mean(audio[i : i + frame] ** 2))))
    if not energies:
        out = os.path.join(out_dir, "0001.wav")
        sf.write(out, audio, sr)
        return [out]

    thr = max(0.01, float(np.median(energies)) * 0.35)
    chunks = []
    start = 0
    last_cut = 0
    max_samples = int(max_sec * sr)
    min_samples = int(min_sec * sr)

    def emit(a, b):
        if b - a < min_samples:
            return
        piece = audio[a:b]
        peak = np.max(np.abs(piece)) or 1.0
        piece = (piece / peak) * 0.95
        name = f"{len(chunks) + 1:04d}.wav"
        out = os.path.join(out_dir, name)
        sf.write(out, piece, sr)
        chunks.append(out)

    i = 0
    while i < len(energies):
        pos = i * hop
        if pos - last_cut >= max_samples:
            emit(last_cut, pos)
            last_cut = pos
        elif energies[i] < thr and pos - last_cut >= min_samples:
            # пауза — режем, если накопилось достаточно
            silence_run = 0
            j = i
            while j < len(energies) and energies[j] < thr:
                silence_run += hop
                j += 1
            if silence_run >= int(0.25 * sr):
                emit(last_cut, pos)
                last_cut = pos + silence_run
                i = j
                continue
        i += 1

    if len(audio) - last_cut >= min_samples:
        emit(last_cut, len(audio))
    if not chunks:
        out = os.path.join(out_dir, "0001.wav")
        sf.write(out, audio, sr)
        chunks.append(out)
    return chunks


def prepare_training_audio(input_paths, work_dir, progress=None):
    raw_dir = os.path.join(work_dir, "raw")
    wav_dir = os.path.join(work_dir, "wav")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)

    prepared = []
    paths = [p for p in input_paths if p and os.path.isfile(p)]
    if not paths:
        raise ValueError("Нет входных аудиофайлов")

    for idx, src in enumerate(paths, 1):
        if progress:
            progress(
                progress=int(5 + (idx - 1) / len(paths) * 15),
                message=f"Подготовка аудио {idx}/{len(paths)}...",
            )
        mono = os.path.join(raw_dir, f"src_{idx:03d}.wav")
        _to_wav_mono(src, mono)
        parts = _split_wav(mono, os.path.join(wav_dir, f"src_{idx:03d}"))
        for p in parts:
            # flatten into wav_dir with unique names
            dest = os.path.join(wav_dir, f"{idx:03d}_{os.path.basename(p)}")
            if os.path.abspath(p) != os.path.abspath(dest):
                shutil.copy2(p, dest)
            prepared.append(dest)

    if len(prepared) < 3:
        raise ValueError(
            f"Слишком мало фрагментов ({len(prepared)}). Нужно больше речи "
            "(лучше 5–10+ минут чистого голоса)."
        )
    return prepared, wav_dir


def transcribe_dataset(wav_paths, work_dir, progress=None):
    from faster_whisper import WhisperModel

    if progress:
        progress(progress=25, message=f"Загрузка Whisper ({WHISPER_MODEL})...")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    meta_path = os.path.join(work_dir, "metadata.csv")
    rows = []
    total = len(wav_paths)
    for idx, path in enumerate(wav_paths, 1):
        if progress:
            progress(
                progress=int(25 + (idx - 1) / total * 35),
                message=f"Распознавание речи {idx}/{total}...",
            )
        segments, info = model.transcribe(path, language="ru", beam_size=1)
        text = " ".join(s.text.strip() for s in segments).strip()
        if len(text) < 2:
            continue
        # piper metadata: filename|text  (relative to audio_dir)
        rows.append((os.path.basename(path), text))

    if len(rows) < 3:
        raise RuntimeError(
            "Whisper почти ничего не распознал. Проверьте качество/язык аудио."
        )

    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        for name, text in rows:
            writer.writerow([name, text])

    return meta_path, rows


def ensure_base_voice(models_dir=None):
    """Скачать русскую базу irina medium (ONNX) для bootstrap/fallback."""
    from huggingface_hub import hf_hub_download

    models_dir = models_dir or os.path.join(os.getcwd(), "models", "piper")
    os.makedirs(models_dir, exist_ok=True)
    onnx = hf_hub_download(
        repo_id=HF_VOICE_REPO,
        filename=f"{HF_VOICE_PATH}.onnx",
        local_dir=models_dir,
    )
    cfg = hf_hub_download(
        repo_id=HF_VOICE_REPO,
        filename=f"{HF_VOICE_PATH}.onnx.json",
        local_dir=models_dir,
    )
    return onnx, cfg


def _try_find_checkpoint(models_dir):
    patterns = [
        os.path.join(models_dir, "**", "*.ckpt"),
        os.path.join(os.getcwd(), "models", "piper-checkpoints", "**", "*.ckpt"),
    ]
    for pattern in patterns:
        found = glob.glob(pattern, recursive=True)
        if found:
            return found[0]
    return None


def download_finetune_checkpoint(models_dir=None):
    """Пытаемся скачать ckpt для fine-tune (может отсутствовать на HF)."""
    models_dir = models_dir or os.path.join(os.getcwd(), "models", "piper-checkpoints")
    os.makedirs(models_dir, exist_ok=True)
    existing = _try_find_checkpoint(models_dir)
    if existing:
        return existing

    from huggingface_hub import hf_hub_download, list_repo_files

    repo = "rhasspy/piper-checkpoints"
    try:
        files = list_repo_files(repo, repo_type="dataset")
    except Exception:
        files = []

    candidates = [
        f
        for f in files
        if f.endswith(".ckpt") and "ru_RU" in f and "irina" in f and "medium" in f
    ]
    if not candidates:
        candidates = [f for f in files if f.endswith(".ckpt") and "lessac" in f and "medium" in f]
    if not candidates:
        return None

    path = hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=candidates[0],
        local_dir=models_dir,
    )
    return path


def _run_piper_train(work_dir, wav_dir, meta_path, speaker_id, epochs, ckpt_path, progress=None):
    py = _piper_python()
    out_dir = os.path.join(work_dir, "lightning_logs")
    config_path = os.path.join(work_dir, "config.json")
    cache_dir = os.path.join(work_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    cmd = [
        py,
        "-m",
        "piper.train",
        "fit",
        f"--data.voice_name={speaker_id}",
        f"--data.csv_path={meta_path}",
        f"--data.audio_dir={wav_dir}",
        "--model.sample_rate=22050",
        "--data.espeak_voice=ru",
        f"--data.cache_dir={cache_dir}",
        f"--data.config_path={config_path}",
        "--data.batch_size=4",
        f"--trainer.max_epochs={epochs}",
        "--trainer.accelerator=cpu",
        "--trainer.devices=1",
    ]
    if ckpt_path and os.path.isfile(ckpt_path):
        cmd.append(f"--ckpt_path={ckpt_path}")

    if progress:
        progress(progress=65, message=f"Обучение Piper ({epochs} эпох, CPU)...")

    log_path = os.path.join(work_dir, "train.log")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=work_dir, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        tail = ""
        if os.path.isfile(log_path):
            with open(log_path, encoding="utf-8", errors="replace") as f:
                tail = f.read()[-2000:]
        raise RuntimeError(
            "piper.train завершился с ошибкой. Установите: "
            "powershell -File scripts/setup_piper.ps1\n" + tail
        )

    # найти последний ckpt
    ckpts = glob.glob(os.path.join(work_dir, "**", "*.ckpt"), recursive=True)
    if not ckpts:
        ckpts = glob.glob(os.path.join(out_dir, "**", "*.ckpt"), recursive=True)
    if not ckpts:
        raise RuntimeError("Чекпоинт обучения не найден")
    ckpts.sort(key=os.path.getmtime)
    return ckpts[-1], config_path if os.path.isfile(config_path) else None


def _export_onnx(ckpt_path, onnx_out, progress=None):
    py = _piper_python()
    if progress:
        progress(progress=88, message="Экспорт ONNX...")
    cmd = [
        py,
        "-m",
        "piper.train.export_onnx",
        f"--checkpoint={ckpt_path}",
        f"--output-file={onnx_out}",
    ]
    # older API: positional args
    alt = [py, "-m", "piper.train.export_onnx", ckpt_path, onnx_out]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        proc = subprocess.run(alt, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Не удалось экспортировать ONNX:\n"
            + (proc.stderr or proc.stdout or "")[-1500:]
        )
    return onnx_out


def _install_base_as_voice(speaker_id, voices_dir, progress=None):
    """Fallback: поставить русскую базу как стартовую модель (если train недоступен)."""
    if progress:
        progress(progress=70, message="Train недоступен — ставлю базовую ru_RU-irina...")
    onnx_src, cfg_src = ensure_base_voice()
    onnx_dst = os.path.join(voices_dir, f"{speaker_id}.onnx")
    cfg_dst = os.path.join(voices_dir, f"{speaker_id}.onnx.json")
    shutil.copy2(onnx_src, onnx_dst)
    shutil.copy2(cfg_src, cfg_dst)
    return onnx_dst, cfg_dst, "base-irina-fallback"


def train_piper_voice(
    speaker_id,
    reference_paths,
    voices_dir=None,
    epochs=DEFAULT_EPOCHS,
    progress=None,
    allow_base_fallback=True,
):
    voices_dir = voices_dir or os.path.join(os.getcwd(), "voices")
    os.makedirs(voices_dir, exist_ok=True)

    work_dir = tempfile.mkdtemp(prefix=f"piper_train_{speaker_id}_")
    try:
        if progress:
            progress(progress=5, message="Подготовка датасета...")
        wav_paths, wav_dir = prepare_training_audio(reference_paths, work_dir, progress=progress)

        if not is_whisper_available():
            raise RuntimeError(
                "faster-whisper не установлен. Запустите: powershell -File scripts/setup_piper.ps1"
            )

        meta_path, rows = transcribe_dataset(wav_paths, work_dir, progress=progress)

        # сохранить копию датасета рядом с голосом
        dataset_dir = os.path.join(voices_dir, speaker_id, "piper_dataset")
        if os.path.isdir(dataset_dir):
            shutil.rmtree(dataset_dir)
        shutil.copytree(wav_dir, os.path.join(dataset_dir, "wav"))
        shutil.copy2(meta_path, os.path.join(dataset_dir, "metadata.csv"))

        onnx_dst = os.path.join(voices_dir, f"{speaker_id}.onnx")
        cfg_dst = os.path.join(voices_dir, f"{speaker_id}.onnx.json")

        train_ok = False
        mode = "finetune"
        try:
            if progress:
                progress(progress=62, message="Поиск checkpoint для fine-tune...")
            ckpt = download_finetune_checkpoint()
            trained_ckpt, config_path = _run_piper_train(
                work_dir, wav_dir, meta_path, speaker_id, epochs, ckpt, progress=progress
            )
            _export_onnx(trained_ckpt, onnx_dst, progress=progress)
            if config_path and os.path.isfile(config_path):
                shutil.copy2(config_path, cfg_dst)
            elif not os.path.isfile(cfg_dst):
                # config рядом с export
                side = onnx_dst + ".json"
                if os.path.isfile(side):
                    shutil.copy2(side, cfg_dst)
                else:
                    # взять конфиг базы
                    _, base_cfg = ensure_base_voice()
                    shutil.copy2(base_cfg, cfg_dst)
            train_ok = True
        except Exception as exc:
            if not allow_base_fallback:
                raise
            # сохраняем ошибку train в лог профиля
            err_path = os.path.join(voices_dir, speaker_id, "train_error.txt")
            os.makedirs(os.path.dirname(err_path), exist_ok=True)
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(str(exc))
            onnx_dst, cfg_dst, mode = _install_base_as_voice(
                speaker_id, voices_dir, progress=progress
            )
            mode = f"base-fallback:{exc}"

        # smoke test
        if progress:
            progress(progress=95, message="Проверка синтеза Piper...")
        from src.PiperEngine import synthesize

        synthesize("Проверка обученного голоса.", onnx_path=onnx_dst)

        return {
            "onnx": onnx_dst,
            "config": cfg_dst,
            "utterances": len(rows),
            "mode": mode,
            "trained": train_ok,
            "dataset_dir": dataset_dir,
        }
    finally:
        # оставляем work_dir при PIPER_KEEP_WORK=1 для отладки
        if os.environ.get("PIPER_KEEP_WORK") != "1":
            shutil.rmtree(work_dir, ignore_errors=True)
