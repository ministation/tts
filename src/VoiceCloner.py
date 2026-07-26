import os

from src.VoiceRegistry import install_clone_voice


def train_voice_from_upload(
    model,
    speaker,
    reference_paths,
    name,
    sex,
    fallback,
    description,
    engine="piper",
    progress=None,
):
    if isinstance(reference_paths, str):
        reference_paths = [reference_paths]

    if progress:
        progress(progress=8, message=f"Сохранение {len(reference_paths)} образцов...")
    paths = install_clone_voice(
        speaker,
        reference_paths,
        name,
        sex,
        fallback,
        description,
        engine=engine,
    )

    if engine == "piper":
        return _train_piper(speaker, name, paths, progress)
    if engine == "xtts":
        return _train_xtts(speaker, name, paths, progress)
    return _train_silero(model, speaker, name, sex, paths, progress)


def _train_piper(speaker, name, paths, progress):
    from src.PiperTrainer import train_piper_voice

    if progress:
        progress(progress=12, message="Обучение Piper (долго на CPU)...")
    result = train_piper_voice(
        speaker,
        paths["references"],
        progress=progress,
    )
    if progress:
        progress(progress=98, message="Готово")
    return {
        "speaker": speaker,
        "name": name,
        "engine": "piper",
        "utterances": result.get("utterances"),
        "mode": result.get("mode"),
        "trained": result.get("trained"),
        "paths": {
            **paths,
            "onnx": result.get("onnx"),
            "config": result.get("config"),
        },
    }


def _train_silero(model, speaker, name, sex, paths, progress):
    from src.SileroVoiceTrainer import train_from_references, install_trained_model

    if progress:
        progress(progress=20, message="Анализ голоса и подбор модели Silero...")
    trained = train_from_references(model, paths["references"], sex=sex, progress=progress)
    if progress:
        progress(progress=90, message=f"Сохранение {speaker}.pt...")
    install_trained_model(
        speaker,
        trained["model_path"],
        paths["voice_model"],
        tmp_dir=trained.get("tmp_dir"),
    )
    if progress:
        progress(progress=95, message="Проверка синтеза...")
    model.apply_tts(
        text="Проверка клонированного голоса.",
        speaker="random",
        voice_path=paths["voice_model"],
        sample_rate=24000,
        put_accent=True,
        put_yo=False,
    )
    return {
        "speaker": speaker,
        "name": name,
        "engine": "silero",
        "similarity": trained.get("similarity"),
        "paths": paths,
    }


def _train_xtts(speaker, name, paths, progress):
    if progress:
        progress(progress=40, message=f"Создание XTTS-модели {speaker}.pt...")
    from src.CloneClient import ensure_running, encode_voice, synthesize

    ensure_running()
    encode_voice(paths["references"], paths["voice_model"], speaker_id=speaker)
    if progress:
        progress(progress=70, message="Проверка клонирования через XTTS...")
    synthesize(
        "Проверка клонированного голоса.",
        voice_model_path=paths["voice_model"],
        speaker_id=speaker,
    )
    if progress:
        progress(progress=95, message="Готово")
    return {
        "speaker": speaker,
        "name": name,
        "engine": "xtts",
        "paths": paths,
    }
