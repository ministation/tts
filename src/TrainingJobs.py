import os
import shutil
import uuid
import threading

_jobs = {}
_lock = threading.Lock()


def create_job():
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "total": 0,
            "message": "Ожидание...",
            "result": None,
            "error": None,
        }
    return job_id


def update_job(job_id, **fields):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def get_job(job_id):
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def run_job(job_id, fn):
    def wrapper():
        try:
            update_job(job_id, status="running", message="Запуск...")
            result = fn(lambda **kw: update_job(job_id, **kw))
            update_job(job_id, status="done", progress=100, message="Готово", result=result)
        except Exception as exc:
            update_job(job_id, status="error", message=str(exc), error=str(exc))

    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()
