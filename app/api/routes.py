"""
app/api/routes.py
==================
All REST API endpoints for the PDF Translator service.

Routes:
  GET  /health                   — Liveness + readiness probe
  GET  /api/v1/providers         — List providers, models, key status
  GET  /api/v1/languages         — List supported languages
  POST /api/v1/translate         — Synchronous translation
  POST /api/v1/translate/async   — Submit async job
  GET  /api/v1/jobs              — List all jobs
  GET  /api/v1/jobs/<id>         — Get job status
  GET  /api/v1/jobs/<id>/download — Download translated PDF
  DELETE /api/v1/jobs/<id>       — Delete job + files
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_file

from app.core.engine import translate_pdf_file
from app.core.job_store import JobStatus, get_job_store
from app.core.providers import ENV_KEYS, PROVIDER_MODELS, PROVIDERS

log = logging.getLogger(__name__)
bp  = Blueprint("api", __name__)

SUPPORTED_LANGUAGES = [
    "Afrikaans", "Albanian", "Arabic", "Armenian", "Bengali", "Bosnian",
    "Bulgarian", "Catalan", "Chinese (Simplified)", "Chinese (Traditional)",
    "Croatian", "Czech", "Danish", "Dutch", "English", "Estonian", "Finnish",
    "French", "German", "Greek", "Gujarati", "Hebrew", "Hindi", "Hungarian",
    "Icelandic", "Indonesian", "Italian", "Japanese", "Kannada", "Korean",
    "Latvian", "Lithuanian", "Macedonian", "Malay", "Malayalam", "Maltese",
    "Marathi", "Norwegian", "Persian", "Polish", "Portuguese", "Punjabi",
    "Romanian", "Russian", "Serbian", "Slovak", "Slovenian", "Spanish",
    "Swahili", "Swedish", "Tamil", "Telugu", "Thai", "Turkish", "Ukrainian",
    "Urdu", "Vietnamese", "Welsh",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cfg():
    return current_app.config


def _store():
    return get_job_store(current_app.config)


def _job_paths(job_id: str) -> tuple[Path, Path]:
    upload_dir = Path(_cfg().get("UPLOAD_DIR", "/tmp"))
    output_dir = Path(_cfg().get("OUTPUT_DIR", "/tmp"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return (
        upload_dir / f"{job_id}_in.pdf",
        output_dir / f"{job_id}_out.pdf",
    )


def _resolve_api_key(provider: str, override: str | None) -> str | None:
    if override:
        return override
    env_var = ENV_KEYS.get(provider, "")
    if not env_var:
        return None
    return _cfg().get(env_var) or os.environ.get(env_var)


def _parse_params(form) -> dict | tuple[str, int]:
    """Parse and validate common translation parameters."""
    target_lang = (form.get("target_lang") or "").strip()
    if not target_lang:
        return "target_lang is required", 400

    provider = (form.get("provider") or _cfg().get("DEFAULT_PROVIDER", "claude")).strip().lower()
    if provider not in PROVIDERS:
        return (
            f"Unknown provider '{provider}'. "
            f"Available: {', '.join(PROVIDERS)}", 400
        )

    try:
        max_workers = int(form.get("max_workers") or _cfg().get("MAX_WORKERS", 6))
        chunk_size  = int(form.get("chunk_size")  or _cfg().get("CHUNK_SIZE", 20))
    except ValueError:
        return "max_workers and chunk_size must be integers", 400

    return {
        "target_lang": target_lang,
        "provider":    provider,
        "model":       form.get("model") or None,
        "api_key":     _resolve_api_key(provider, form.get("api_key")),
        "source_lang": form.get("source_lang") or None,
        "max_workers": max(1, min(max_workers, 20)),
        "chunk_size":  max(1, min(chunk_size, 50)),
    }


def _safe_unlink(path: Path):
    try:
        if path and path.exists():
            path.unlink()
    except Exception:
        pass


def _run_async_job(app, job_id: str, input_path: Path, output_path: Path, params: dict):
    """Background thread worker."""
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
                status       = JobStatus.DONE,
                finished_at  = time.time(),
                output_path  = str(output_path),
                result       = result,
            )
            log.info(f"Job {job_id[:8]} done: {result['spans']} spans, {result['size_mb']} MB")

        except Exception as exc:
            log.error(f"Job {job_id[:8]} failed: {exc}", exc_info=True)
            store.update(
                job_id,
                status      = JobStatus.FAILED,
                finished_at = time.time(),
                error       = str(exc),
            )
        finally:
            _safe_unlink(input_path)


# ── Health ────────────────────────────────────────────────────────────────────

@bp.get("/health")
def health():
    """Liveness + readiness probe."""
    store   = _store()
    counts  = store.count_by_status()
    # Purge expired jobs on health check
    try:
        store.purge_expired()
    except Exception:
        pass
    return jsonify({
        "status":  "ok",
        "service": "pdf-translator",
        "version": "1.0.0",
        "jobs":    counts,
    })


# ── Providers ────────────────────────────────────────────────────────────────

@bp.get("/api/v1/providers")
def list_providers():
    cfg     = _cfg()
    result  = []
    for name, cls in PROVIDERS.items():
        env_var    = ENV_KEYS.get(name, "")
        key_set    = bool(cfg.get(env_var) or os.environ.get(env_var, "")) if env_var else True
        result.append({
            "provider":       name,
            "default_model":  cls.default_model,
            "models":         PROVIDER_MODELS.get(name, []),
            "env_key":        env_var or "not required",
            "key_configured": key_set,
        })
    return jsonify({"providers": result, "count": len(result)})


# ── Languages ────────────────────────────────────────────────────────────────

@bp.get("/api/v1/languages")
def list_languages():
    return jsonify({"languages": SUPPORTED_LANGUAGES, "count": len(SUPPORTED_LANGUAGES)})


# ── Synchronous translate ─────────────────────────────────────────────────────

@bp.post("/api/v1/translate")
def translate_sync():
    """
    Upload PDF → get translated PDF in response.

    Form fields:
      file         (required) — PDF binary upload
      target_lang  (required) — target language, e.g. "French"
      provider     (optional) — claude|openai|gemini|grok|groq|mistral|cohere|ollama
      model        (optional) — override model name
      api_key      (optional) — override env-var API key
      source_lang  (optional) — skip auto-detection
      max_workers  (optional) — parallel threads (1–20, default 6)
      chunk_size   (optional) — spans per LLM call (1–50, default 20)

    Response: application/pdf with X-* metadata headers
    """
    if "file" not in request.files:
        return jsonify(error="No 'file' field in request"), 400

    f = request.files["file"]
    if not f.filename or not f.filename.lower().endswith(".pdf"):
        return jsonify(error="File must be a PDF (.pdf extension required)"), 400

    params = _parse_params(request.form)
    if isinstance(params, tuple):
        return jsonify(error=params[0]), params[1]

    job_id = "sync-" + os.urandom(8).hex()
    input_path, output_path = _job_paths(job_id)

    try:
        f.save(str(input_path))
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
            config        = _cfg(),
        )

        stem      = Path(f.filename).stem
        lang_tag  = params["target_lang"].lower().replace(" ", "_").replace("(", "").replace(")", "")
        dl_name   = f"{stem}_{lang_tag}.pdf"

        response = send_file(
            str(output_path),
            mimetype      = "application/pdf",
            as_attachment = True,
            download_name = dl_name,
        )
        response.headers["X-Pages"]       = str(result.get("pages", ""))
        response.headers["X-Spans"]       = str(result.get("spans", ""))
        response.headers["X-Provider"]    = result.get("provider", "")
        response.headers["X-Model"]       = result.get("model", "")
        response.headers["X-Source-Lang"] = result.get("source_lang", "")
        response.headers["X-Target-Lang"] = result.get("target_lang", "")
        response.headers["X-Size-MB"]     = str(result.get("size_mb", ""))
        return response

    except FileNotFoundError as e:
        return jsonify(error=str(e)), 404
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        log.error(f"Sync translate error: {e}", exc_info=True)
        return jsonify(error=f"Translation failed: {str(e)}"), 500
    finally:
        _safe_unlink(input_path)
        # output_path cleaned after response is sent via a background task
        # For simplicity in production use object storage and pre-signed URLs


# ── Async translate ───────────────────────────────────────────────────────────

@bp.post("/api/v1/translate/async")
def translate_async():
    """
    Submit a translation job. Returns job_id immediately.
    Poll GET /api/v1/jobs/<job_id> — download when status is "done".
    """
    if "file" not in request.files:
        return jsonify(error="No 'file' field in request"), 400

    f = request.files["file"]
    if not f.filename or not f.filename.lower().endswith(".pdf"):
        return jsonify(error="File must be a PDF (.pdf extension required)"), 400

    params = _parse_params(request.form)
    if isinstance(params, tuple):
        return jsonify(error=params[0]), params[1]

    store = _store()

    # Check concurrent job cap
    counts = store.count_by_status()
    active = counts.get("queued", 0) + counts.get("processing", 0)
    max_concurrent = _cfg().get("MAX_CONCURRENT_JOBS", 20)
    if active >= max_concurrent:
        return jsonify(
            error=f"Server busy ({active} active jobs). Please retry shortly."
        ), 503

    job_id = store.create(
        target_lang = params["target_lang"],
        provider    = params["provider"],
        model       = params.get("model") or "",
        filename    = f.filename,
    )

    input_path, output_path = _job_paths(job_id)
    f.save(str(input_path))
    store.update(job_id, input_path=str(input_path), output_path=str(output_path))

    threading.Thread(
        target  = _run_async_job,
        args    = (current_app._get_current_object(), job_id, input_path, output_path, params),
        daemon  = True,
    ).start()

    base = request.host_url.rstrip("/")
    return jsonify({
        "job_id":       job_id,
        "status":       JobStatus.QUEUED,
        "poll_url":     f"{base}/api/v1/jobs/{job_id}",
        "download_url": f"{base}/api/v1/jobs/{job_id}/download",
    }), 202


# ── Jobs ─────────────────────────────────────────────────────────────────────

@bp.get("/api/v1/jobs")
def list_jobs():
    store = _store()
    jobs  = store.list_all()
    base  = request.host_url.rstrip("/")
    slim  = [
        {
            "job_id":      j.get("job_id"),
            "status":      j.get("status"),
            "provider":    j.get("provider"),
            "target_lang": j.get("target_lang"),
            "created_at":  j.get("created_at"),
            "finished_at": j.get("finished_at"),
            "download_url": f"{base}/api/v1/jobs/{j.get('job_id')}/download"
                            if j.get("status") == JobStatus.DONE else None,
        }
        for j in jobs
    ]
    return jsonify({"count": len(slim), "jobs": slim})


@bp.get("/api/v1/jobs/<job_id>")
def get_job(job_id: str):
    store = _store()
    job   = store.get(job_id)
    if not job:
        return jsonify(error="Job not found"), 404

    resp = {
        "job_id":      job.get("job_id"),
        "status":      job.get("status"),
        "provider":    job.get("provider"),
        "model":       job.get("model") or None,
        "target_lang": job.get("target_lang"),
        "filename":    job.get("filename") or None,
        "created_at":  job.get("created_at"),
        "started_at":  job.get("started_at") or None,
        "finished_at": job.get("finished_at") or None,
        "error":       job.get("error") or None,
    }

    # Duration
    if job.get("created_at") and job.get("finished_at"):
        try:
            resp["duration_seconds"] = round(float(job["finished_at"]) - float(job["created_at"]), 2)
        except Exception:
            pass

    # Result metadata
    raw_result = job.get("result")
    if raw_result and job.get("status") == JobStatus.DONE:
        result = raw_result if isinstance(raw_result, dict) else {}
        resp["result"] = {
            "pages":       result.get("pages"),
            "spans":       result.get("spans"),
            "source_lang": result.get("source_lang"),
            "size_mb":     result.get("size_mb"),
            "warning":     result.get("warning"),
        }
        base = request.host_url.rstrip("/")
        resp["download_url"] = f"{base}/api/v1/jobs/{job_id}/download"

    return jsonify(resp)


@bp.get("/api/v1/jobs/<job_id>/download")
def download_job(job_id: str):
    store = _store()
    job   = store.get(job_id)
    if not job:
        return jsonify(error="Job not found"), 404

    status = job.get("status")
    if status in (JobStatus.QUEUED, JobStatus.PROCESSING):
        return jsonify(error=f"Job is still {status}. Poll /api/v1/jobs/{job_id} and retry."), 202
    if status == JobStatus.FAILED:
        return jsonify(error=f"Job failed: {job.get('error')}"), 500
    if status != JobStatus.DONE:
        return jsonify(error=f"Unexpected job status: {status}"), 500

    output_path = job.get("output_path")
    if not output_path or not Path(output_path).exists():
        return jsonify(error="Output file not found or has expired"), 404

    lang_tag = (job.get("target_lang") or "translated").lower().replace(" ", "_")
    return send_file(
        output_path,
        mimetype      = "application/pdf",
        as_attachment = True,
        download_name = f"translated_{lang_tag}.pdf",
    )


@bp.delete("/api/v1/jobs/<job_id>")
def delete_job(job_id: str):
    store = _store()
    job   = store.get(job_id)
    if not job:
        return jsonify(error="Job not found"), 404

    # Prevent deleting running jobs
    if job.get("status") in (JobStatus.QUEUED, JobStatus.PROCESSING):
        return jsonify(error="Cannot delete an active job"), 409

    _safe_unlink(Path(job.get("input_path") or ""))
    _safe_unlink(Path(job.get("output_path") or ""))
    store.delete(job_id)
    return jsonify({"deleted": job_id})
