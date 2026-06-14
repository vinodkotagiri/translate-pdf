"""
app/core/tasks.py
==================
Celery task definitions for async PDF translation jobs.

Requires REDIS_URL to be set (same env var used by the job store).
The worker is launched via:
  celery -A app.core.tasks.celery_app worker --loglevel=info --concurrency=2
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from celery import Celery

log = logging.getLogger(__name__)

_redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "pdf_tasks",
    broker  = _redis_url,
    backend = _redis_url,
)

celery_app.conf.update(
    task_serializer        = "json",
    result_serializer      = "json",
    accept_content         = ["json"],
    task_track_started     = True,
    worker_prefetch_multiplier = 1,   # one task per worker at a time (PDF jobs are heavy)
    task_acks_late         = True,    # ack only after the task finishes (safe for long jobs)
)


@celery_app.task(name="translate_pdf_task")
def translate_pdf_task(
    job_id:      str,
    input_path:  str,
    output_path: str,
    params:      dict,
) -> None:
    """
    Celery task that mirrors the threading._run_async_job logic from routes.py.
    Creates its own Flask app context so it can access config and the job store.
    """
    from app import create_app
    from app.core.engine import translate_pdf_file
    from app.core.job_store import JobStatus, get_job_store

    app = create_app()
    with app.app_context():
        store = get_job_store(app.config)
        try:
            store.update(job_id, status=JobStatus.PROCESSING, started_at=time.time())

            result = translate_pdf_file(
                input_path    = input_path,
                output_path   = output_path,
                target_lang   = params["target_lang"],
                provider_name = params["provider"],
                api_key       = params["api_key"],
                model         = params["model"],
                max_workers   = params["max_workers"],
                chunk_size    = params["chunk_size"],
                source_lang   = params["source_lang"],
                config        = app.config,
            )

            store.update(
                job_id,
                status      = JobStatus.DONE,
                finished_at = time.time(),
                output_path = str(output_path),
                result      = result,
            )
            log.info("Celery job %s done: %s spans, %s MB",
                     job_id[:8], result["spans"], result["size_mb"])

        except Exception as exc:
            log.error("Celery job %s failed: %s", job_id[:8], exc, exc_info=True)
            store.update(
                job_id,
                status      = JobStatus.FAILED,
                finished_at = time.time(),
                error       = str(exc),
            )
        finally:
            p = Path(input_path)
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
