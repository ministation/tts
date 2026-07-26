#!/usr/bin/env bash
# Установка SS14 TTS API в .venv на Linux-хосте.
# Целевая директория по умолчанию: /home/ss14_user/ss14_tts_api
set -euo pipefail

APP_DIR="${APP_DIR:-/home/ss14_user/ss14_tts_api}"
PYTHON_BIN="${PYTHON_BIN:-python3.9}"
VENV_DIR="${APP_DIR}/.venv"

if [[ ! -d "${APP_DIR}" ]]; then
  echo "Каталог не найден: ${APP_DIR}"
  echo "Склонируйте репозиторий туда или задайте APP_DIR=..."
  exit 1
fi

cd "${APP_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "Нужен Python 3.9+ (рекомендуется 3.9–3.11)"
    exit 1
  fi
fi

echo "==> Система: espeak-ng, libsndfile (для Piper/Silero)"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    espeak-ng libsndfile1 ffmpeg git build-essential
fi

echo "==> Python: ${PYTHON_BIN} ($("${PYTHON_BIN}" -V))"
if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools
echo "==> Зависимости (CPU torch)"
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-prod.txt

echo "==> Голоса Piper + Silero"
python tools/prepare_production.py

echo ""
echo "Готово."
echo "  Активация: source ${VENV_DIR}/bin/activate"
echo "  Ручной запуск: cd ${APP_DIR} && .venv/bin/python wsgi.py"
echo "  systemd: см. deploy/ss14-tts.service и docs/DEPLOY.ru.md"
