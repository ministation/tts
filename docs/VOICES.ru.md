# Как добавлять голоса в SS14 TTS API

Два основных режима:

1. **Встроенные / Silero** — быстрые голоса `aidar`, `baya`, `kseniya`, `xenia`, `eugene` и алиасы.
2. **Piper ONNX (рекомендуется для своих голосов)** — долгое офлайн-обучение по вашему аудио, быстрый синтез на CPU.

XTTS остаётся как legacy (медленный рантайм).

## Рекомендуемый путь: обучение Piper

```powershell
powershell -File scripts/setup_piper.ps1
# нужен espeak-ng: winget install eSpeak-NG.eSpeak-NG
python ss14tts.py
```

Откройте http://127.0.0.1:5000/train

1. Выберите движок **Piper**.
2. Загрузите **как можно больше** чистой речи одного человека (лучше **5–10+ минут**, до 50 файлов).
3. Дождитесь пайплайна: подготовка → Whisper → fine-tune → `voices/<id>.onnx`.
4. Обучение на CPU может занять **часы** — это нормально.
5. После обучения `/tts` с этим `speaker` идёт через Piper ONNX (быстро).

Структура после обучения:

```
voices/
  config.yml
  ivan.onnx
  ivan.onnx.json
  ivan/
    references/
    piper_dataset/
      wav/
      metadata.csv
```

## Как устроен рантайм

```
POST /tts { speaker, text }
  ├─ engine: piper  → voices/<id>.onnx  (CPU ONNX, быстро)
  ├─ engine: silero → Silero model.pt / voices/<id>.pt
  └─ engine: xtts   → legacy clone_worker (медленно)
```

Игровой контракт SS14 не меняется: тот же `speaker` в `tts-voices.yml`.

## Silero: встроенные и random `.pt`

```bash
python tools/voice_manager.py list
python tools/voice_manager.py generate --count 10
python tools/voice_manager.py preview --file voices/tmp/Voice3.pt
python tools/voice_manager.py install captain voices/tmp/Voice3.pt --name "Капитан" --sex Male --fallback eugene
```

1. **Есть `voices/<id>.onnx` + `engine: piper`** → Piper.
2. **Есть `voices/<id>.pt` + `engine: silero`** → кастомный Silero.
3. **Нет модели, есть запись в `config.yml`** → `fallback` builtin.
4. **Алиасы** (arthas, pudge…) → маппинг в `SpeakerPatch`.

## Подключение к игре SS14

Файл `tts-voices.yml` — прототипы для клиента. Скопируйте в `/Resources/Prototypes/Corvax` и добавьте локализацию.

```yaml
- type: ttsVoice
  id: Ivan
  name: tts-voice-name-ivan
  sex: Male
  speaker: ivan
  roundStart: false
```

Поле `speaker` должно совпадать с ID API (`ivan` → `voices/ivan.onnx`).

```ftl
tts-voice-name-ivan = Иван
```

## API

```bash
# Статус обучающего стека
curl http://127.0.0.1:5000/voices/train/status

# Список голосов
curl "http://127.0.0.1:5000/voices?detailed=1"

# Синтез
curl -X POST http://127.0.0.1:5000/tts \
  -H "Content-Type: application/json" \
  -d "{\"api_token\":\"test\",\"speaker\":\"ivan\",\"text\":\"Привет\",\"format\":\"ogg\"}"
```

## Советы

- Для Piper давайте **много речи**, не 30 секунд.
- Первый запуск Whisper/Piper скачает модели.
- Переменная `PIPER_TRAIN_EPOCHS` (по умолчанию 200) — число эпох обучения.
- `PIPER_KEEP_WORK=1` оставляет временную папку обучения для отладки.
- XTTS worker (`clone_worker.py`) для новых голосов **не нужен**.

## Docker

Смонтируйте папку голосов:

```bash
docker run -d --name ss14tts -p 5000:5000 \
  -v /path/to/my/voices:/workspace/voices \
  -e apitoken=YOUR_TOKEN \
  backmen/ss14-tts:latest
```
