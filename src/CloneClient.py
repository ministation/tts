import os
import subprocess
import time

import requests

CLONE_URL = os.environ.get("CLONE_URL", "http://127.0.0.1:5001")
CLONE_PYTHON = os.environ.get(
    "CLONE_PYTHON",
    os.path.join(os.getcwd(), "clone_venv", "Scripts", "python.exe"),
)
_worker_proc = None
_log_handle = None
_session = requests.Session()
_session.trust_env = False


def _worker_script():
    return os.path.join(os.getcwd(), "clone_worker.py")


def _log_path():
    return os.path.join(os.getcwd(), "clone_worker.log")


def is_available():
    return os.path.isfile(CLONE_PYTHON) and os.path.isfile(_worker_script())


def health(timeout=2):
    try:
        r = _session.get(f"{CLONE_URL}/health", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _port_busy():
    import socket

    host = CLONE_URL.replace("http://", "").replace("https://", "").split("/")[0]
    if ":" in host:
        hostname, port_s = host.split(":", 1)
        port = int(port_s)
    else:
        hostname, port = host, 80
    if hostname in ("127.0.0.1", "localhost"):
        hostname = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((hostname, port)) == 0


def start_worker():
    global _worker_proc, _log_handle
    if health():
        return True
    if not is_available():
        raise RuntimeError(
            "XTTS не установлен. Запустите: powershell -File scripts/setup_clone.ps1"
        )

    if _worker_proc is not None and _worker_proc.poll() is None:
        return False

    if _port_busy() and health():
        return True

    if _port_busy():
        return False

    _log_handle = open(_log_path(), "a", encoding="utf-8")
    _log_handle.write(f"\n--- starting worker {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    _log_handle.flush()
    _worker_proc = subprocess.Popen(
        [CLONE_PYTHON, _worker_script()],
        cwd=os.getcwd(),
        stdout=_log_handle,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    return False


def ensure_running(wait_seconds=180):
    if health():
        return True

    start_worker()
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if health(timeout=3):
            return True
        if _worker_proc is not None and _worker_proc.poll() is not None:
            tail = ""
            if os.path.isfile(_log_path()):
                with open(_log_path(), encoding="utf-8", errors="replace") as f:
                    tail = f.read()[-1500:]
            raise RuntimeError(
                "XTTS worker упал при запуске. Смотрите clone_worker.log\n" + tail
            )
        time.sleep(2)

    raise RuntimeError(
        "XTTS worker не отвечает. Откройте второй терминал и выполните:\n"
        "  clone_venv\\Scripts\\python.exe clone_worker.py\n"
        f"Лог: {_log_path()}"
    )


def warmup(timeout=600):
    ensure_running()
    r = _session.post(f"{CLONE_URL}/warmup", timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"XTTS warmup failed: {r.text}")


def synthesize(text, reference_path, language="ru", timeout=600):
    ensure_running()
    r = _session.post(
        f"{CLONE_URL}/synthesize",
        json={"text": text, "reference": os.path.abspath(reference_path), "language": language},
        timeout=timeout,
    )
    if r.status_code != 200:
        try:
            msg = r.json().get("description", r.text)
        except Exception:
            msg = r.text
        raise RuntimeError(f"XTTS error: {msg}")
    return r.json()
