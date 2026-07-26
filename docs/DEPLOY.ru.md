# Деплой SS14 TTS API на Linux (systemd + .venv)

Целевая директория: `/home/ss14_user/ss14_tts_api`

В продакшене только **Silero** (builtin + роли + 16 custom `.pt`) и **Piper** (Denis, Dmitri, Irina, Ruslan). XTTS и клоны удалены.

## 1. Пользователь и каталог

```bash
sudo useradd -m -s /bin/bash ss14_user || true
sudo mkdir -p /home/ss14_user/ss14_tts_api
# скопируйте код репозитория в этот каталог (git clone / rsync / scp)
sudo chown -R ss14_user:ss14_user /home/ss14_user/ss14_tts_api
```

Пример копирования с вашей машины:

```bash
rsync -av --exclude .venv --exclude piper_venv --exclude clone_venv --exclude __pycache__ \
  ./ ss14_user@HOST:/home/ss14_user/ss14_tts_api/
```

## 2. Установка .venv и голосов

От имени `ss14_user` (или с `sudo -u ss14_user`):

```bash
cd /home/ss14_user/ss14_tts_api
chmod +x scripts/setup_linux.sh
./scripts/setup_linux.sh
```

Скрипт:
- ставит `espeak-ng`, `libsndfile1`, `ffmpeg`
- создаёт `.venv`
- ставит CPU-torch + `requirements-prod.txt`
- чистит лишние голоса и качает Piper/Silero через `tools/prepare_production.py`

Переменные (опционально):

```bash
APP_DIR=/home/ss14_user/ss14_tts_api PYTHON_BIN=python3.11 ./scripts/setup_linux.sh
```

Рекомендуется Python **3.9–3.11**.

## 3. Токен и порт

По умолчанию: **порт `8000`**, **apitoken `test`** (уже в `deploy/ss14-tts.service`).

В конфиге игры:

```toml
[tts]
api_url = "http://TTS_HOST:8000/tts"
api_token = "test"
enabled = true
```

## 4. systemd

```bash
sudo cp /home/ss14_user/ss14_tts_api/deploy/ss14-tts.service /etc/systemd/system/ss14-tts.service
sudo systemctl daemon-reload
sudo systemctl enable --now ss14-tts
sudo systemctl status ss14-tts
journalctl -u ss14-tts -f
```

Проверка:

```bash
curl -s http://127.0.0.1:8000/health
curl -s 'http://127.0.0.1:8000/voices?detailed=1' | head
```

Веб-тестер: `http://HOST:8000/`

## 5. Файлы для SS14

Скопируйте в ресурсы сервера игры:

- `tts-voices.yml`
- `tts-voices-localization.ftl` → `Resources/Locale/ru-RU/tts/tts-voices.ftl` (или ваш путь)

## 6. Обновление

```bash
cd /home/ss14_user/ss14_tts_api
# git pull / rsync новых файлов
source .venv/bin/activate
pip install -r requirements-prod.txt
python tools/prepare_production.py
sudo systemctl restart ss14-tts
```

## Состав голосов

| Движок | ID |
|--------|-----|
| Silero builtin | aidar, baya, kseniya, xenia, eugene |
| Silero роли | oleg, zina, garithos, maiev, myron, narrator, captain, doctor, bartender, scientist, security, clown |
| Piper | denis, dmitri, irina, ruslan |
| Silero custom | viktor … sofia (16 шт.) |
