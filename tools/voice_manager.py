#!/usr/bin/env python3
"""
Утилита управления голосами SS14 TTS.

Примеры:
  python tools/voice_manager.py list
  python tools/voice_manager.py curate --count 15
  python tools/voice_manager.py preview --file voices/tmp/curate/001.pt
  python tools/voice_manager.py install captain voices/tmp/curate/001.pt --name "Капитан" --sex Male
  python tools/voice_manager.py use-builtin captain
"""

import argparse
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
WARMUP_PASSES = 3

EXAMPLE_TEXTS = [
    "Капитан, доложите обстановку на мостике.",
    "Внимание! Обнаружена утечка плазмы в инженерном отсеке.",
    "Медицинский отсек открыт. Если вам плохо — приходите, мы поможем.",
    "Эс Бэ! Тут человек в сером костюме, с тулбоксом и в маске!",
    "Добро пожаловать на станцию. Соблюдайте технику безопасности.",
    "Клоун, прекрати разбрасывать банановые кожурки офицерам под ноги!",
]


def load_model():
    import torch

    local_file = "model.pt"
    if not os.path.isfile(local_file):
        print("Скачиваю модель Silero v3_1_ru...")
        torch.hub.download_url_to_file(
            "https://models.silero.ai/models/tts/ru/v3_1_ru.pt",
            local_file,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)
    return model


def _warmup_and_save(model, path):
    for _ in range(WARMUP_PASSES):
        model.apply_tts(
            text=random.choice(EXAMPLE_TEXTS),
            speaker="random",
            sample_rate=SAMPLE_RATE,
            put_accent=True,
            put_yo=False,
        )
    model.save_random_voice(path)


def cmd_list(args):
    from src.VoiceRegistry import list_all_voices

    voices = list_all_voices(["aidar", "baya", "kseniya", "xenia", "eugene", "random"])
    print(f"{'ID':<14} {'Режим':<8} {'Играет':<10} Имя")
    print("-" * 62)
    for v in voices:
        mode = v.get("source", "builtin")
        active = v.get("active_speaker") or "-"
        print(f"{v['id']:<14} {mode:<8} {active:<10} {v['name']}")


def cmd_generate(args):
    model = load_model()
    out_dir = args.output or os.path.join("voices", "tmp")
    os.makedirs(out_dir, exist_ok=True)
    prefix = args.prefix or "Voice"

    print(f"Генерирую {args.count} кандидатов в {out_dir}/")
    print("Совет: для отбора на слух используйте curate — он сразу создаёт WAV-превью.")
    for idx in range(1, args.count + 1):
        path = os.path.join(out_dir, f"{prefix}{idx}.pt")
        _warmup_and_save(model, path)
        print(f"  [{idx}/{args.count}] {path}")
    print(f"\npython tools/voice_manager.py preview --file {out_dir}/{prefix}1.pt")


def cmd_curate(args):
    import soundfile as sf

    model = load_model()
    out_dir = args.output or os.path.join("voices", "tmp", "curate")
    os.makedirs(out_dir, exist_ok=True)
    text = args.text or "Капитан, доложите обстановку. Всему персоналу занять боевые посты."

    print(f"Создаю {args.count} кандидатов с WAV-превью в {out_dir}/")
    print("Слушайте .wav файлы и выберите лучший. Пол не гарантирован — проверяйте на слух!")
    print()

    for idx in range(1, args.count + 1):
        pt_path = os.path.join(out_dir, f"{idx:03d}.pt")
        wav_path = os.path.join(out_dir, f"{idx:03d}.wav")
        _warmup_and_save(model, pt_path)
        audio = model.apply_tts(
            text=text,
            speaker="random",
            voice_path=pt_path,
            sample_rate=SAMPLE_RATE,
            put_accent=True,
            put_yo=False,
        )
        sf.write(wav_path, audio, SAMPLE_RATE)
        print(f"  [{idx:03d}] {wav_path}")

    print()
    print("Установить понравившийся (переключит голос в режим custom):")
    print(f"  python tools/voice_manager.py install captain {out_dir}/005.pt --name Капитан --sex Male --fallback eugene")


def cmd_preview(args):
    import soundfile as sf

    model = load_model()
    voice_path = args.file
    if not os.path.isfile(voice_path):
        print(f"Файл не найден: {voice_path}")
        sys.exit(1)
    text = args.text or random.choice(EXAMPLE_TEXTS)
    audio = model.apply_tts(
        text=text,
        speaker="random",
        voice_path=voice_path,
        sample_rate=SAMPLE_RATE,
        put_accent=True,
        put_yo=False,
    )
    out = args.output or "preview.wav"
    sf.write(out, audio, SAMPLE_RATE)
    print(f"Текст: {text}")
    print(f"Сохранено: {out}")


def cmd_install(args):
    from src.VoiceRegistry import add_voice_to_config

    speaker = args.speaker
    src = args.file
    if not os.path.isfile(src):
        print(f"Файл не найден: {src}")
        sys.exit(1)
    if not speaker.isidentifier() or speaker != speaker.lower():
        print("speaker должен быть в нижнем регистре латиницей (например: oleg, captain)")
        sys.exit(1)
    dst = os.path.join("voices", f"{speaker}.pt")
    shutil.copy2(src, dst)
    print(f"Установлено: {dst}")
        add_voice_to_config(
            speaker,
            args.name or speaker,
            args.sex,
            args.fallback,
            args.description or "",
            source="custom",
            engine="silero",
        )
    print("Режим: custom (используется .pt). Вернуть чистый Silero: use-builtin")


def cmd_use_builtin(args):
    from src.VoiceRegistry import set_voice_source

    set_voice_source(args.speaker, "builtin")
    print(f"Голос '{args.speaker}' переключён на builtin (чистый Silero fallback)")


def cmd_uninstall(args):
    from src.VoiceRegistry import set_voice_source

    path = os.path.join("voices", f"{args.speaker}.pt")
    if os.path.isfile(path):
        os.remove(path)
        print(f"Удалено: {path}")
    try:
        set_voice_source(args.speaker, "builtin")
        print(f"Голос '{args.speaker}' переключён на builtin")
    except KeyError:
        print(f"Запись '{args.speaker}' не найдена в config.yml")


def cmd_register(args):
    from src.VoiceRegistry import add_voice_to_config

    add_voice_to_config(
        args.speaker,
        args.name,
        args.sex,
        args.fallback,
        args.description or "",
        source="builtin",
    )
    print(f"Голос '{args.speaker}' зарегистрирован (режим builtin)")


def main():
    parser = argparse.ArgumentParser(description="Управление голосами SS14 TTS")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Список всех голосов")

    gen = sub.add_parser("generate", help="Сгенерировать .pt кандидатов")
    gen.add_argument("--count", type=int, default=10)
    gen.add_argument("--output", help="Папка для .pt")
    gen.add_argument("--prefix", default="Voice")

    cur = sub.add_parser("curate", help="Сгенерировать кандидатов + WAV для отбора на слух")
    cur.add_argument("--count", type=int, default=15)
    cur.add_argument("--output", help="Папка вывода (по умолчанию voices/tmp/curate)")
    cur.add_argument("--text", help="Текст для превью")

    prev = sub.add_parser("preview", help="Прослушать .pt (сохраняет WAV)")
    prev.add_argument("--file", required=True)
    prev.add_argument("--text")
    prev.add_argument("--output", default="preview.wav")

    inst = sub.add_parser("install", help="Установить .pt и включить режим custom")
    inst.add_argument("speaker")
    inst.add_argument("file")
    inst.add_argument("--name")
    inst.add_argument("--sex", choices=["Male", "Female", "Unsexed"], default="Unsexed")
    inst.add_argument("--fallback")
    inst.add_argument("--description")

    ub = sub.add_parser("use-builtin", help="Вернуть голос на чистый Silero")
    ub.add_argument("speaker")

    un = sub.add_parser("uninstall", help="Удалить .pt и вернуть builtin")
    un.add_argument("speaker")

    reg = sub.add_parser("register", help="Добавить запись в config.yml (builtin)")
    reg.add_argument("speaker")
    reg.add_argument("--name", required=True)
    reg.add_argument("--sex", choices=["Male", "Female", "Unsexed"], default="Unsexed")
    reg.add_argument("--fallback")
    reg.add_argument("--description")

    args = parser.parse_args()
    {
        "list": cmd_list,
        "generate": cmd_generate,
        "curate": cmd_curate,
        "preview": cmd_preview,
        "install": cmd_install,
        "use-builtin": cmd_use_builtin,
        "uninstall": cmd_uninstall,
        "register": cmd_register,
    }[args.command](args)


if __name__ == "__main__":
    main()
