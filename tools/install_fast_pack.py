#!/usr/bin/env python3
"""
Установка пакета из 20 быстрых голосов для SS14 TTS.

- 4× Piper RU (ONNX, лучшее качество среди открытых RU на CPU)
- 16× Silero custom .pt (быстрый синтез, уникальный тембр)

Запуск:
  python tools/install_fast_pack.py
"""
from __future__ import annotations

import os
import random
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SAMPLE_RATE = 24000
HF_REPO = "rhasspy/piper-voices"

# Официальные русские Piper (все доступные medium)
PIPER_VOICES = [
    {
        "id": "denis",
        "name": "Денис",
        "sex": "Male",
        "fallback": "eugene",
        "hf": "ru/ru_RU/denis/medium/ru_RU-denis-medium",
        "description": "Piper RU Denis — мужской, чёткий",
    },
    {
        "id": "dmitri",
        "name": "Дмитрий",
        "sex": "Male",
        "fallback": "aidar",
        "hf": "ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium",
        "description": "Piper RU Dmitri — мужской, спокойный",
    },
    {
        "id": "irina",
        "name": "Ирина",
        "sex": "Female",
        "fallback": "xenia",
        "hf": "ru/ru_RU/irina/medium/ru_RU-irina-medium",
        "description": "Piper RU Irina — женский, нейтральный",
    },
    {
        "id": "ruslan",
        "name": "Руслан",
        "sex": "Male",
        "fallback": "eugene",
        "hf": "ru/ru_RU/ruslan/medium/ru_RU-ruslan-medium",
        "description": "Piper RU Ruslan — мужской, низкий",
    },
]

# Дополнительные быстрые Silero .pt под роли станции
SILERO_VOICES = [
    {"id": "viktor", "name": "Виктор", "sex": "Male", "fallback": "eugene", "description": "Silero custom — мужской"},
    {"id": "andrei", "name": "Андрей", "sex": "Male", "fallback": "aidar", "description": "Silero custom — мужской"},
    {"id": "sergei", "name": "Сергей", "sex": "Male", "fallback": "eugene", "description": "Silero custom — мужской"},
    {"id": "pavel", "name": "Павел", "sex": "Male", "fallback": "aidar", "description": "Silero custom — мужской"},
    {"id": "nikita", "name": "Никита", "sex": "Male", "fallback": "eugene", "description": "Silero custom — мужской"},
    {"id": "roman", "name": "Роман", "sex": "Male", "fallback": "aidar", "description": "Silero custom — мужской"},
    {"id": "anton", "name": "Антон", "sex": "Female", "fallback": "xenia", "description": "Silero custom — женский"},
    {"id": "maxim", "name": "Максим", "sex": "Male", "fallback": "aidar", "description": "Silero custom — мужской"},
    {"id": "anna", "name": "Анна", "sex": "Female", "fallback": "xenia", "description": "Silero custom — женский"},
    {"id": "maria", "name": "Мария", "sex": "Female", "fallback": "kseniya", "description": "Silero custom — женский"},
    {"id": "elena", "name": "Елена", "sex": "Female", "fallback": "xenia", "description": "Silero custom — женский"},
    {"id": "olga", "name": "Ольга", "sex": "Female", "fallback": "kseniya", "description": "Silero custom — женский"},
    {"id": "natalia", "name": "Наталья", "sex": "Male", "fallback": "eugene", "description": "Silero custom — мужской"},
    {"id": "svetlana", "name": "Светлана", "sex": "Female", "fallback": "kseniya", "description": "Silero custom — женский"},
    {"id": "daria", "name": "Дарья", "sex": "Female", "fallback": "xenia", "description": "Silero custom — женский"},
    {"id": "sofia", "name": "София", "sex": "Female", "fallback": "kseniya", "description": "Silero custom — женский"},
]

WARMUP = [
    "Капитан, доложите обстановку на мостике.",
    "Внимание! Обнаружена утечка плазмы в инженерном отсеке.",
    "Медицинский отсек открыт. Если вам плохо — приходите.",
    "Добро пожаловать на станцию. Соблюдайте технику безопасности.",
]


def download_piper(voice: dict):
    from huggingface_hub import hf_hub_download

    from src.VoiceRegistry import add_voice_to_config

    voice_id = voice["id"]
    onnx_dst = os.path.join("voices", f"{voice_id}.onnx")
    cfg_dst = os.path.join("voices", f"{voice_id}.onnx.json")
    os.makedirs("voices", exist_ok=True)

    if not os.path.isfile(onnx_dst):
        print(f"  скачиваю Piper {voice_id}...")
        onnx_src = hf_hub_download(
            repo_id=HF_REPO,
            filename=f"{voice['hf']}.onnx",
            local_dir=os.path.join("models", "piper"),
        )
        cfg_src = hf_hub_download(
            repo_id=HF_REPO,
            filename=f"{voice['hf']}.onnx.json",
            local_dir=os.path.join("models", "piper"),
        )
        shutil.copy2(onnx_src, onnx_dst)
        shutil.copy2(cfg_src, cfg_dst)
    else:
        print(f"  уже есть {onnx_dst}")

    add_voice_to_config(
        voice_id,
        voice["name"],
        voice["sex"],
        voice["fallback"],
        voice["description"],
        source="custom",
        engine="piper",
    )
    return voice_id


def load_silero():
    import torch

    local_file = "model.pt"
    if not os.path.isfile(local_file):
        print("Скачиваю Silero v3_1_ru...")
        torch.hub.download_url_to_file(
            "https://models.silero.ai/models/tts/ru/v3_1_ru.pt",
            local_file,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)
    return model


def make_silero_voice(model, voice: dict):
    from src.VoiceRegistry import add_voice_to_config

    voice_id = voice["id"]
    pt_path = os.path.join("voices", f"{voice_id}.pt")
    os.makedirs("voices", exist_ok=True)

    if not os.path.isfile(pt_path):
        print(f"  генерирую Silero {voice_id}...")
        for _ in range(3):
            model.apply_tts(
                text=random.choice(WARMUP),
                speaker="random",
                sample_rate=SAMPLE_RATE,
                put_accent=True,
                put_yo=False,
            )
        model.save_random_voice(pt_path)
    else:
        print(f"  уже есть {pt_path}")

    add_voice_to_config(
        voice_id,
        voice["name"],
        voice["sex"],
        voice["fallback"],
        voice["description"],
        source="custom",
        engine="silero",
    )
    return voice_id


def sync_game_files(voice_ids: list):
    """Дописать tts-voices.yml и localization, не дублируя существующие speaker."""
    from src.VoiceRegistry import get_voice_config, load_config

    yml_path = "tts-voices.yml"
    ftl_path = "tts-voices-localization.ftl"

    existing_yml = open(yml_path, encoding="utf-8").read() if os.path.isfile(yml_path) else ""
    existing_ftl = open(ftl_path, encoding="utf-8").read() if os.path.isfile(ftl_path) else ""

    yml_add = []
    ftl_add = []
    for vid in voice_ids:
        entry = get_voice_config(vid) or {}
        name = entry.get("name", vid)
        sex = entry.get("sex", "Unsexed")
        proto_id = "".join(p.capitalize() for p in vid.split("_"))
        if f"speaker: {vid}" not in existing_yml:
            block = (
                f"\n- type: ttsVoice\n"
                f"  id: {proto_id}\n"
                f"  name: tts-voice-name-{vid}\n"
                f"  sex: {sex}\n"
                f"  speaker: {vid}\n"
            )
            yml_add.append(block)
        key = f"tts-voice-name-{vid}"
        if key not in existing_ftl:
            ftl_add.append(f"{key} = {name}\n")

    if yml_add:
        with open(yml_path, "a", encoding="utf-8") as f:
            f.write("\n# --- fast pack (auto) ---\n")
            f.writelines(yml_add)
        print(f"Обновлён {yml_path} (+{len(yml_add)})")
    if ftl_add:
        with open(ftl_path, "a", encoding="utf-8") as f:
            f.write("\n# --- fast pack (auto) ---\n")
            f.writelines(ftl_add)
        print(f"Обновлён {ftl_path} (+{len(ftl_add)})")


def main():
    print("=== Пакет 20 быстрых голосов ===\n")
    print("1/2 Piper RU (4 голоса)...")
    installed = []
    for v in PIPER_VOICES:
        installed.append(download_piper(v))

    print("\n2/2 Silero custom (16 голосов)...")
    model = load_silero()
    for v in SILERO_VOICES:
        installed.append(make_silero_voice(model, v))

    print("\nСинхронизация файлов для SS14...")
    sync_game_files(installed)

    print("\nГотово! Голоса:")
    for vid in installed:
        print(f"  - {vid}")
    print("\nПроверка: python ss14tts.py → http://127.0.0.1:5000/")
    print("В игру: скопируйте tts-voices.yml и tts-voices-localization.ftl")
    print("Гайд: docs/GUIDE.ru.md")


if __name__ == "__main__":
    main()
