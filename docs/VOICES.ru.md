# Как добавлять голоса в SS14 TTS API

Движок: [Silero TTS v3.1 (русский)](https://models.silero.ai/). Есть 5 встроенных голосов (`aidar`, `baya`, `kseniya`, `xenia`, `eugene`) и произвольные кастомные голоса в формате `.pt`.

## Быстрый старт

```bash
# Список всех голосов (с пометкой, есть ли .pt модель)
python tools/voice_manager.py list

# Сгенерировать 10 кандидатов
python tools/voice_manager.py generate --count 10

# Прослушать кандидата (сохранит preview.wav)
python tools/voice_manager.py preview --file voices/tmp/Voice3.pt

# Установить понравившийся голос
python tools/voice_manager.py install captain voices/tmp/Voice3.pt --name "Капитан" --sex Male --fallback eugene
```

После установки перезапуск сервера **не нужен** — API подхватывает файлы сразу.

## Как это устроено

```
voices/
  config.yml      ← реестр голосов (имя, пол, fallback)
  oleg.pt         ← кастомная модель голоса (Silero random voice)
  captain.pt
  tmp/            ← черновики при генерации
```

1. **Есть `voices/<id>.pt`** → используется уникальный кастомный голос (лучшее качество, каждый .pt — свой тембр).
2. **Нет .pt, но есть запись в `config.yml`** → играет через `fallback` (встроенный Silero).
3. **Нет ни того, ни другого** → старые алиасы (arthas, pudge…) мапятся на встроенные голоса; неизвестные → `baya`.

## Способ 1: CLI (рекомендуется)

### Шаг 1 — сгенерировать кандидатов

```bash
python tools/voice_manager.py generate --count 20 --output voices/tmp
```

Каждый запуск `speaker="random"` + `save_random_voice()` создаёт новый уникальный тембр.

### Шаг 2 — выбрать лучший

```bash
python tools/voice_manager.py preview --file voices/tmp/Voice5.pt --text "Привет, это тест голоса капитана!"
```

Откройте `preview.wav` в плеере. Повторите для разных номеров.

### Шаг 3 — установить

```bash
python tools/voice_manager.py install captain voices/tmp/Voice5.pt \
  --name "Капитан" --sex Male --fallback eugene --description "Командный голос"
```

Команда копирует `.pt` в `voices/captain.pt` и обновляет `config.yml`.

### Только регистрация (без .pt, будет fallback)

```bash
python tools/voice_manager.py register bartender --name "Бармен" --sex Male --fallback eugene
```

## Способ 2: Вручную скопировать .pt

1. Получите файл `myvoice.pt` (из генератора или ноутбука `tts_test.ipynb`).
2. Положите в `voices/myvoice.pt` — имя файла = speaker ID для API.
3. Добавьте метаданные в `voices/config.yml`:

```yaml
voices:
  myvoice:
    name: Мой голос
    sex: Male
    fallback: aidar
    description: Описание для себя
```

## Способ 3: HTTP API (загрузка)

```bash
curl -X POST http://127.0.0.1:5000/voices/upload \
  -H "Content-Type: application/json" \
  -d "{
    \"api_token\": \"test\",
    \"speaker\": \"captain\",
    \"file\": \"<base64 содержимое captain.pt>\",
    \"register\": true,
    \"name\": \"Капитан\",
    \"sex\": \"Male\",
    \"fallback\": \"eugene\"
  }"
```

Поле `register: true` добавит запись в `config.yml`.

## Способ 4: Jupyter ноутбук

Откройте `tts_test.ipynb`:

1. Установите зависимости (ячейка с pip).
2. Запустите генератор (`Generate(model, 20)`).
3. Прослушайте голоса в ноутбуке.
4. Переименуйте понравившийся: `voices/tmp/Voice7.pt` → `voices/captain.pt`.

## Подключение к игре SS14

Файл `tts-voices.yml` — прототипы для клиента игры. Скопируйте в `/Resources/Prototypes/Corvax` и добавьте локализацию.

Пример для нового голоса:

```yaml
- type: ttsVoice
  id: Captain
  name: tts-voice-name-captain
  sex: Male
  speaker: captain
```

Поле `speaker` **должно совпадать** с ID в API (`captain` → `voices/captain.pt`).

Добавьте в `.ftl` локализацию:

```ftl
tts-voice-name-captain = Капитан
```

## Docker

Смонтируйте папку голосов, чтобы они сохранялись между перезапусками:

```bash
docker run -d --name ss14tts -p 5000:5000 \
  -v /path/to/my/voices:/workspace/voices \
  -e apitoken=YOUR_TOKEN \
  backmen/ss14-tts:latest
```

## Советы по качеству

- Генерируйте **15–30 кандидатов** и выбирайте на слух — у random голосов сильно разный тембр.
- Для прослушки используйте **типичные SS14 фразы** (см. `EXAMPLE_TEXTS` в `tools/voice_manager.py`).
- Кастомный `.pt` всегда лучше, чем fallback на встроенный голос.
- ID голоса: только **латиница нижним регистром**, цифры и `_` (например `captain`, `doc_2`).
- Встроенные fallback: `aidar`/`eugene` (м), `kseniya`/`xenia` (ж), `baya` (унисекс).

## API: список голосов

```bash
# Простой список ID
curl http://127.0.0.1:5000/voices

# Подробности (имя, пол, есть ли модель, fallback)
curl "http://127.0.0.1:5000/voices?detailed=1"
```

## Веб-тестер

Откройте `http://127.0.0.1:5000/` — список голосов подтягивается автоматически с метаданными.
