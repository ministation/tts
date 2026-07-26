# Деплой SS14 TTS API на Linux (systemd + .venv)

Целевая директория: **`/home/ss14_user/ss14_tts_api`**

## Одна команда от root

Код уже должен лежать в каталоге API, затем:

```bash
cd /home/ss14_user/ss14_tts_api && bash scripts/install_root.sh
```

Скрипт сам: apt-пакеты → `.venv` → голоса → systemd (`User=root`) → `enable --now` → проверка `/health`.

Порт **8000**, токен **test**. Переопределение:

```bash
PORT=8000 apitoken=test bash scripts/install_root.sh
```

## Ручная установка (если нужно)

```bash
cd /home/ss14_user/ss14_tts_api
bash scripts/setup_linux.sh
cp deploy/ss14-tts.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now ss14-tts
curl -s http://127.0.0.1:8000/health
```

## Токен и порт

По умолчанию: **порт `8000`**, **apitoken `test`**.

В конфиге игры:

```toml
[tts]
api_url = "http://TTS_HOST:8000/tts"
api_token = "test"
enabled = true
```

## Логи / статус

```bash
systemctl status ss14-tts
journalctl -u ss14-tts -f
curl -s http://127.0.0.1:8000/health
curl -s 'http://127.0.0.1:8000/voices?detailed=1' | head
```

Веб-тестер: `http://HOST:8000/`

## Файлы для SS14

- `tts-voices.yml`
- `tts-voices-localization.ftl` → `Resources/Locale/ru-RU/tts/tts-voices.ftl`

## Обновление

```bash
cd /home/ss14_user/ss14_tts_api
# git pull / rsync
bash scripts/install_root.sh
```

## Состав голосов

| Движок | ID |
|--------|-----|
| Silero builtin | aidar, baya, kseniya, xenia, eugene |
| Silero роли | oleg, zina, garithos, maiev, myron, narrator, captain, doctor, bartender, scientist, security, clown |
| Piper | denis, dmitri, irina, ruslan |
| Silero custom | viktor … sofia (16 шт.) |
