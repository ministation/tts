# ss14-tts-api

### сборка своего образа
```
docker build . -t ss14-tts-api:latest
docker run -it -d --name ss14tts -p 5000:5000 ss14-tts-api:latest
```

### запуск в докере
```
docker run -it -d --name ss14tts -p 5000:5000 backmen/ss14-tts:latest
```

### удаление (для обновления)
```
docker rm -f ss14tts
```

просмотр логов в докере:
```
docker logs ss14tts -f
```

### запуск без docker

windows: https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe

!!! выше версии 3.9 не работет !!!

```
pip3 install -r ./requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
python ss14tts.py
```

### Прочие

конфиг:

```
[tts]
api_url="http://127.0.0.1:8000/tts"
api_token="test"
enabled=true
```


файл с бесплатными голосами:
> tts-voices.yml

копирнуть в /Resources/Prototypes/Corvax

### Продакшен на Linux (systemd + .venv)

Каталог: `/home/ss14_user/ss14_tts_api`  
Только Silero + Piper. Инструкция: **[docs/DEPLOY.ru.md](docs/DEPLOY.ru.md)**

```bash
cd /home/ss14_user/ss14_tts_api && bash scripts/install_root.sh
# порт 8000, apitoken=test, сервис ss14-tts от root
```

### Добавление своих голосов

**Полный гайд:** [docs/GUIDE.ru.md](docs/GUIDE.ru.md)

**Пакет 20 быстрых голосов одной командой:**
```powershell
python -m pip install -r requirements.txt
scripts\setup_piper.bat
python tools/install_fast_pack.py
python ss14tts.py
```

Свои клоны (долгое обучение): http://127.0.0.1:5000/train

Для Docker смонтируйте volume: `-v /path/to/voices:/workspace/voices`

### Скрипт быстрого запуска

```
#!/bin/bash
# Использование готового Docker-образа
docker_image_name="backmen/ss14-tts:latest"

# Переменные окружения
threads=$(nproc)  # Количество ядер процессора
apitoken="YOUR_API_TOKEN"  # Здесь укажите свой секретный ключ

# Запуск Docker-образа с авто-перезагрузкой и публикацией порта 5000
container_name="ss14-tts-api-container"
docker pull "$docker_image_name" >/dev/null 2>&1
docker stop "$container_name" >/dev/null 2>&1
docker rm "$container_name" >/dev/null 2>&1
docker run -d \
    --name "$container_name" \
    -p 5000:5000 \
    --restart always \
    -e "threads=$threads" \
    -e "apitoken=$apitoken" \
    "$docker_image_name"

# Получение внешнего IP-адреса
external_ip=$(curl -s ifconfig.me)

# Вывод конфигурационных параметров
echo "[tts]"
echo "api_url=\"http://$external_ip:5000/tts\""
echo "api_token=\"$apitoken\""
echo "enabled=true"
```
