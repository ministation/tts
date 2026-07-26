#!/usr/bin/env python3
"""
Подготовка продакшен-голосов: только Piper + Silero.

Удаляет XTTS/экспериментальные профили, ставит пакет быстрых голосов.

  python tools/prepare_production.py
"""
from __future__ import annotations

import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

KEEP_IDS = {
    # Piper
    "denis", "dmitri", "irina", "ruslan",
    # Silero custom
    "viktor", "andrei", "sergei", "pavel", "nikita", "roman", "anton", "maxim",
    "anna", "maria", "elena", "olga", "natalia", "svetlana", "daria", "sofia",
}

KEEP_FILES = {
    "config.yml",
}
for vid in KEEP_IDS:
    KEEP_FILES.add(f"{vid}.pt")
    KEEP_FILES.add(f"{vid}.onnx")
    KEEP_FILES.add(f"{vid}.onnx.json")
    KEEP_FILES.add(f"{vid}.json")


def clean_voices_dir():
    voices = os.path.join(ROOT, "voices")
    os.makedirs(voices, exist_ok=True)
    removed = []
    for name in os.listdir(voices):
        path = os.path.join(voices, name)
        if name in KEEP_FILES:
            continue
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            removed.append(name + "/")
        else:
            try:
                os.remove(path)
                removed.append(name)
            except OSError as exc:
                print(f"  не удалось удалить {name}: {exc}")
    return removed


def main():
    print("=== Продакшен: очистка лишних голосов ===")
    removed = clean_voices_dir()
    if removed:
        print(f"Удалено ({len(removed)}): {', '.join(sorted(removed)[:20])}"
              + ("..." if len(removed) > 20 else ""))
    else:
        print("Лишних файлов нет.")

    # config.yml / tts-voices уже в репо; докачиваем модели
    print("\n=== Установка Piper + Silero пакета ===")
    import runpy
    runpy.run_path(os.path.join(ROOT, "tools", "install_fast_pack.py"), run_name="__main__")
    print("\nГотово. Продакшен-голоса: Piper(4) + Silero custom(16) + builtin/роли.")


if __name__ == "__main__":
    main()
