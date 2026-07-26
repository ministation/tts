# Полный гайд: SS14 TTS (быстрые голоса на CPU)

## Что получите

| Тип | Скорость в игре | Сходство с «вашим» голосом | Когда использовать |
|---|---|---|---|
| **Piper ONNX** (готовые RU) | очень быстро | фиксированные дикторы | основной пакет |
| **Silero .pt** (custom) | очень быстро | уникальный тембр, не клон | доп. персонажи |
| **Piper train** с вашего аудио | очень быстро после обучения | близко к вам (нужно много речи) | свой голос |
| **XTTS** (legacy) | медленно | сильный клон | не для продакшена |

---

## Шаг 0. Один раз установить

```powershell
cd C:\ss14\tts
python -m pip install -r requirements.txt
scripts\setup_piper.bat
winget install eSpeak-NG.eSpeak-NG
```

Если `.ps1` блокируется политикой:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_piper.ps1
```

---

## Шаг 1. Поставить 20 готовых быстрых голосов (авто)

```powershell
cd C:\ss14\tts
python tools/install_fast_pack.py
```

Скрипт:

1. Скачает **4 русских Piper**: `denis`, `dmitri`, `irina`, `ruslan`
2. Сгенерирует **16 Silero** `.pt`: `viktor`…`sofia`
3. Пропишет их в `voices/config.yml`
4. Допишет `tts-voices.yml` и `tts-voices-localization.ftl`

> Открытых качественных русских Piper-моделей официально только **4**. Остальные 16 — быстрые уникальные Silero. Это максимум «идеально+быстро» без дней обучения.

---

## Шаг 2. Запустить API

```powershell
python ss14tts.py
```

- Тестер: http://127.0.0.1:5000/
- Обучение своего голоса: http://127.0.0.1:5000/train

---

## Шаг 3. Подключить к сборке SS14

### 3.1. Прототипы голосов

Скопируйте обновлённый [`tts-voices.yml`](../tts-voices.yml) в:

```
Resources/Prototypes/Corvax/tts-voices.yml
```

(или смержите блоки с `speaker: denis`, `irina`, …)

### 3.2. Локализация

Строки из [`tts-voices-localization.ftl`](../tts-voices-localization.ftl) → например:

```
Resources/Locale/ru-RU/tts/tts-voices.ftl
```

### 3.3. Конфиг сервера/клиента

```
[tts]
api_url="http://IP_СЕРВЕРА_TTS:5000/tts"
api_token="test"
enabled=true
```

На проде смените `apitoken` (переменная окружения `apitoken` у TTS).

### 3.4. На Ubuntu рядом с игрой

```bash
sudo apt install -y python3.9 python3.9-venv ffmpeg espeak-ng
cd /opt/ss14-tts
python3.9 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# скопируйте папку voices/ с Windows (onnx + pt + config.yml)
python ss14tts.py
```

---

## Свой голос «как я» (не готовый пакет)

### Откуда брать аудио

**Лучше всего — своя запись:**

1. Тихая комната, один микрофон, без музыки
2. Читайте разные фразы 10–30+ минут (новости, диалоги, роли)
3. Формат WAV/MP3, один человек
4. Можно несколько файлов

**Легальные открытые корпуса (для тестов / чужих тембров):**

| Источник | Ссылка | Заметка |
|---|---|---|
| Mozilla Common Voice (RU) | https://commonvoice.mozilla.org/ru/datasets | много спикеров, нужна нарезка по одному человеку |
| OpenSLR / public datasets | https://www.openslr.org/ | ищите русские корпусы |
| LibriVox (RU книги) | https://librivox.org/ | только если лицензия позволяет ваш сервер |

Не качайте случайные TikTok/YouTube без прав — для публичного сервера это риск.

### Как обучить

1. http://127.0.0.1:5000/train  
2. Движок **Piper**  
3. Загрузить 5–10+ минут речи  
4. Ждать часы на CPU  
5. Получите `voices/<id>.onnx` — синтез быстрый  

Либо CLI-пайплайн уже встроен в `/voices/train`.

---

## Список голосов из пакета

**Piper (fast, лучшее качество RU):**

- `denis`, `dmitri`, `irina`, `ruslan`

**Silero custom (fast):**

- м: `viktor`, `andrei`, `sergei`, `pavel`, `nikita`, `roman`, `anton`, `maxim`
- ж: `anna`, `maria`, `elena`, `olga`, `natalia`, `svetlana`, `daria`, `sofia`

Плюс старые builtin/алиасы: `aidar`, `captain`, `oleg`, …

---

## Частые проблемы

| Ошибка | Решение |
|---|---|
| `Execution Policy` на `.ps1` | `scripts\setup_piper.bat` или `-ExecutionPolicy Bypass` |
| `No module named stressrnn` | `python -m pip install -r requirements.txt` (не `requirments`) |
| Piper train падает | всё равно останется датасет; поставьте `piper_venv` через `setup_piper.bat` |
| В игре нет голоса | `speaker` в yml ≠ id в `config.yml` / нет файла onnx\|pt |
| Медленно | это XTTS — для игры используйте Piper/Silero |

---

## Честно про «идеальные»

- **Идеально быстро + открытый RU** → Piper `irina/denis/dmitri/ruslan`
- **Идеально «мой голос»** → долгое Piper-обучение на **ваших** 10+ минутах
- **20 уникальных клонов знаменитостей из интернета** → не автоматизируем (права + этика)

Готовый автопакет даёт **20 быстрых игровых голосов прямо сейчас**. Свой тембр — через `/train`.
