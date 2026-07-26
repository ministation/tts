#!/usr/bin/env bash
# Одна команда от root: зависимости + .venv + голоса + systemd + старт.
#
#   cd /home/ss14_user/ss14_tts_api && bash scripts/install_root.sh
#
# Или если код уже в APP_DIR:
#   bash /home/ss14_user/ss14_tts_api/scripts/install_root.sh
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Запускайте от root: sudo bash scripts/install_root.sh"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${APP_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${APP_DIR}/.venv"
SERVICE_NAME="ss14-tts"
APITOKEN="${apitoken:-test}"
PORT="${PORT:-8000}"

cd "${APP_DIR}"

if [[ ! -f "${APP_DIR}/wsgi.py" ]]; then
  echo "Не найден wsgi.py в ${APP_DIR}"
  echo "Сначала положите код API в этот каталог."
  exit 1
fi

echo "==> APP_DIR=${APP_DIR}"

echo "==> apt пакеты"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  python3 python3-venv python3-pip \
  espeak-ng libsndfile1 ffmpeg git build-essential curl

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi
echo "==> Python: ${PYTHON_BIN} ($("${PYTHON_BIN}" -V))"

echo "==> .venv"
if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools
echo "==> torch (CPU) + requirements-prod"
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r "${APP_DIR}/requirements-prod.txt"

echo "==> голоса Piper + Silero"
python "${APP_DIR}/tools/prepare_production.py"

echo "==> systemd unit (root, port ${PORT}, token ${APITOKEN})"
cat > "/etc/systemd/system/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=SS14 TTS API (Silero + Piper)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=${APP_DIR}
Environment=apitoken=${APITOKEN}
Environment=PORT=${PORT}
Environment=threads=6
Environment=PATH=${VENV_DIR}/bin:/usr/local/bin:/usr/bin
ExecStart=${VENV_DIR}/bin/python ${APP_DIR}/wsgi.py
Restart=on-failure
RestartSec=5
TimeoutStartSec=600
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ss14-tts
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

# эталон в репо тоже под root
cp "/etc/systemd/system/${SERVICE_NAME}.service" "${APP_DIR}/deploy/ss14-tts.service"

systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}"

echo "==> ждём health..."
ok=0
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    ok=1
    break
  fi
  sleep 2
done

systemctl --no-pager --full status "${SERVICE_NAME}" || true
if [[ "${ok}" -eq 1 ]]; then
  echo ""
  echo "OK: http://127.0.0.1:${PORT}/health"
  curl -s "http://127.0.0.1:${PORT}/health" || true
  echo ""
else
  echo ""
  echo "Сервис не ответил на /health. Логи:"
  journalctl -u "${SERVICE_NAME}" -n 80 --no-pager || true
  exit 1
fi
