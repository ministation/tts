import os
import shutil
import tempfile

from src.VoiceRegistry import install_clone_voice


def clone_voice_from_upload(speaker, reference_path, name, sex, fallback, description, progress=None):
    if progress:
        progress(progress=20, message="Сохранение голосового профиля...")
    paths = install_clone_voice(speaker, reference_path, name, sex, fallback, description)
    if progress:
        progress(progress=60, message="Проверка клонирования через XTTS...")
    from src.CloneClient import ensure_running, synthesize

    ensure_running()
    try:
        from src.CloneClient import warmup
        warmup(timeout=600)
    except Exception:
        pass
    synthesize("Проверка клонированного голоса.", paths["reference"])
    if progress:
        progress(progress=95, message="Готово")
    return {
        "speaker": speaker,
        "name": name,
        "engine": "xtts",
        "paths": paths,
    }
