import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

SAMPLE_RATE = 24000
MAX_SECONDS = 30
MAX_SECONDS_PER_FILE = 15
MAX_REFERENCE_FILES = 5


def _ffmpeg_path():
    cwd = os.getcwd()
    for name in ("ffmpeg.exe", "ffmpeg"):
        local = os.path.join(cwd, "bin", name)
        if os.path.isfile(local):
            return local
    return "ffmpeg"


def prepare_reference(input_path, output_path, max_seconds=MAX_SECONDS, target_sr=SAMPLE_RATE):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = [
        _ffmpeg_path(),
        "-y",
        "-i",
        input_path,
        "-ar",
        str(target_sr),
        "-ac",
        "1",
        "-t",
        str(max_seconds),
        "-af",
        "loudnorm",
        output_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode == 0 and os.path.isfile(output_path):
            return output_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    import librosa

    audio, _ = librosa.load(input_path, sr=target_sr, mono=True, duration=max_seconds)
    peak = np.max(np.abs(audio)) or 1.0
    audio = (audio / peak) * 0.95
    sf.write(output_path, audio, target_sr)
    return output_path


def prepare_references(input_paths, output_dir, max_files=MAX_REFERENCE_FILES):
    os.makedirs(output_dir, exist_ok=True)
    paths = [p for p in input_paths if p and os.path.isfile(p)][:max_files]
    if not paths:
        raise ValueError("No input audio files")
    max_seconds = MAX_SECONDS_PER_FILE if len(paths) > 1 else MAX_SECONDS
    prepared = []
    for idx, input_path in enumerate(paths, 1):
        output_path = os.path.join(output_dir, f"{idx:03d}.wav")
        prepare_reference(input_path, output_path, max_seconds=max_seconds)
        prepared.append(output_path)
    return prepared


def temp_upload_path(upload_storage, suffix=".upload"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    upload_storage.save(path)
    return path
