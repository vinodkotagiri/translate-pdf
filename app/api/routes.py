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

from flask import Blueprint, Response, current_app, jsonify, request, send_file

from app.core.engine import translate_pdf_file
from app.core.job_store import JobStatus, get_job_store
from app.core.providers import ENV_KEYS, PROVIDER_MODELS, PROVIDERS

# Celery is optional — falls back to threading when not installed or broker unreachable
_celery_task = None
try:
    from app.core.tasks import translate_pdf_task as _celery_task
except ImportError:
    pass

log = logging.getLogger(__name__)
bp  = Blueprint("api", __name__)

SUPPORTED_LANGUAGES = [
    # ── Indian languages (22 official scheduled languages + Sanskrit) ──────────
    "Hindi", "Bengali", "Telugu", "Marathi", "Tamil", "Urdu", "Gujarati",
    "Kannada", "Malayalam", "Odia", "Punjabi", "Assamese", "Maithili",
    "Sanskrit", "Kashmiri", "Sindhi", "Konkani", "Dogri", "Bodo", "Nepali",
    # ── Other Asian ───────────────────────────────────────────────────────────
    "Arabic", "Persian", "Hebrew", "Thai", "Japanese",
    "Chinese (Simplified)", "Chinese (Traditional)", "Korean", "Vietnamese",
    "Indonesian", "Malay",
    # ── European ─────────────────────────────────────────────────────────────
    "Afrikaans", "Albanian", "Armenian", "Bosnian", "Bulgarian", "Catalan",
    "Croatian", "Czech", "Danish", "Dutch", "English", "Estonian", "Finnish",
    "French", "German", "Greek", "Hungarian", "Icelandic", "Italian",
    "Latvian", "Lithuanian", "Macedonian", "Maltese", "Norwegian", "Polish",
    "Portuguese", "Romanian", "Russian", "Serbian", "Slovak", "Slovenian",
    "Spanish", "Swahili", "Swedish", "Turkish", "Ukrainian", "Welsh",
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
    """Parse and validate translation parameters. Only target_lang comes from the caller;
    everything else is taken from server configuration."""
    target_lang = (form.get("target_lang") or "").strip()
    if not target_lang:
        return "target_lang is required", 400

    provider = _cfg().get("DEFAULT_PROVIDER", "openai").strip().lower()

    return {
        "target_lang": target_lang,
        "provider":    provider,
        "model":       None,                              # use provider default
        "api_key":     _resolve_api_key(provider, None),  # always from env var
        "source_lang": None,                              # auto-detect
        "max_workers": int(_cfg().get("MAX_WORKERS", 6)),
        "chunk_size":  int(_cfg().get("CHUNK_SIZE", 20)),
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


# ── API Documentation ─────────────────────────────────────────────────────────

@bp.get("/api/docs")
def api_docs():
    """Serve Swagger UI for interactive API exploration."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>PDF Translator API Docs</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css"/>
  <style>
    body { margin: 0; }
    #swagger-ui .topbar { background: #1a1a2e; }
    #swagger-ui .topbar-wrapper img { content: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6zm-1 1.5L18.5 9H13V3.5zM12 17v-4m0 0V9m0 4h-3m3 0h3"/></svg>'); height:32px; }
  </style>
</head>
<body>
<div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>
  SwaggerUIBundle({
    url: "/api/openapi.json",
    dom_id: "#swagger-ui",
    presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
    layout: "BaseLayout",
    deepLinking: true,
    displayOperationId: false,
    defaultModelsExpandDepth: 1,
    defaultModelExpandDepth: 1,
  });
</script>
</body>
</html>"""
    return Response(html, mimetype="text/html")


@bp.get("/api/openapi.json")
def openapi_spec():
    """Return the OpenAPI 3.0 specification for this API."""
    base = request.host_url.rstrip("/")
    spec = {
        "openapi": "3.0.3",
        "info": {
            "title": "PDF Translator API",
            "version": "1.0.0",
            "description": (
                "Translate any PDF to any language while preserving all layout, images, "
                "font sizes, colors, and formatting.\n\n"
                "**Just two fields required**: upload your PDF and pick a target language — "
                "everything else is handled automatically.\n\n"
                "Supports all 22 Indian scheduled languages plus 40+ world languages. "
                "Maximum 500 pages per document."
            ),
            "contact": {"name": "API Support", "email": "support@example.com"},
        },
        "servers": [{"url": base, "description": "This server"}],
        "tags": [
            {"name": "translation", "description": "Submit and retrieve PDF translations"},
            {"name": "jobs",        "description": "Async job lifecycle management"},
            {"name": "metadata",    "description": "Providers, languages, and health"},
        ],
        "paths": {
            "/health": {
                "get": {
                    "tags": ["metadata"],
                    "summary": "Health check",
                    "description": "Liveness + readiness probe. Also purges expired jobs.",
                    "operationId": "health",
                    "responses": {
                        "200": {
                            "description": "Service is healthy",
                            "content": {"application/json": {"schema": {
                                "type": "object",
                                "properties": {
                                    "status":  {"type": "string", "example": "ok"},
                                    "service": {"type": "string", "example": "pdf-translator"},
                                    "version": {"type": "string", "example": "1.0.0"},
                                    "jobs":    {"type": "object", "description": "Job counts by status"},
                                },
                            }}},
                        }
                    },
                }
            },
            "/api/v1/providers": {
                "get": {
                    "tags": ["metadata"],
                    "summary": "List available LLM providers",
                    "operationId": "listProviders",
                    "responses": {
                        "200": {
                            "description": "Provider list",
                            "content": {"application/json": {"schema": {
                                "type": "object",
                                "properties": {
                                    "count":     {"type": "integer"},
                                    "providers": {"type": "array", "items": {
                                        "type": "object",
                                        "properties": {
                                            "provider":        {"type": "string"},
                                            "default_model":   {"type": "string"},
                                            "models":          {"type": "array", "items": {"type": "string"}},
                                            "env_key":         {"type": "string"},
                                            "key_configured":  {"type": "boolean"},
                                        },
                                    }},
                                },
                            }}},
                        }
                    },
                }
            },
            "/api/v1/languages": {
                "get": {
                    "tags": ["metadata"],
                    "summary": "List supported target languages",
                    "description": "Indian languages appear first and are mandatory.",
                    "operationId": "listLanguages",
                    "responses": {
                        "200": {
                            "description": "Language list",
                            "content": {"application/json": {"schema": {
                                "type": "object",
                                "properties": {
                                    "count":     {"type": "integer"},
                                    "languages": {"type": "array", "items": {"type": "string"},
                                                  "example": ["Hindi", "Bengali", "Tamil", "English"]},
                                },
                            }}},
                        }
                    },
                }
            },
            "/api/v1/translate": {
                "post": {
                    "tags": ["translation"],
                    "summary": "Translate PDF (synchronous)",
                    "description": (
                        "Upload a PDF and receive the translated PDF in the response body. "
                        "The call blocks until translation is complete — use the async endpoint "
                        "for documents larger than ~20 pages.\n\n"
                        "Original text is deleted from the PDF content stream (not just painted "
                        "over), so the output is clean and fully searchable in the target language."
                    ),
                    "operationId": "translateSync",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "required": ["file", "target_lang"],
                                    "properties": {
                                        "file": {
                                            "type": "string",
                                            "format": "binary",
                                            "description": "PDF file to translate (max 100 MB, 500 pages)",
                                        },
                                        "target_lang": {
                                            "type": "string",
                                            "example": "Hindi",
                                            "description": "Target language — e.g. Hindi, Tamil, French, Japanese",
                                        },
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Translated PDF",
                            "headers": {
                                "X-Pages":       {"schema": {"type": "integer"}, "description": "Page count"},
                                "X-Spans":       {"schema": {"type": "integer"}, "description": "Translated span count"},
                                "X-Provider":    {"schema": {"type": "string"},  "description": "LLM provider used"},
                                "X-Model":       {"schema": {"type": "string"},  "description": "Model used"},
                                "X-Source-Lang": {"schema": {"type": "string"},  "description": "Detected source language"},
                                "X-Target-Lang": {"schema": {"type": "string"},  "description": "Target language"},
                                "X-Size-MB":     {"schema": {"type": "number"},  "description": "Output file size"},
                            },
                            "content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}},
                        },
                        "400": {"description": "Validation error"},
                        "413": {"description": "File exceeds upload size limit"},
                        "422": {"description": "Document exceeds 500-page limit"},
                        "500": {"description": "Translation failed"},
                    },
                }
            },
            "/api/v1/translate/async": {
                "post": {
                    "tags": ["translation"],
                    "summary": "Translate PDF (asynchronous)",
                    "description": (
                        "Submit a translation job and get a `job_id` back immediately. "
                        "Recommended for documents larger than ~20 pages.\n\n"
                        "**Workflow**: submit → poll `GET /api/v1/jobs/{job_id}` until "
                        "`status` is `done` → download from `GET /api/v1/jobs/{job_id}/download`."
                    ),
                    "operationId": "translateAsync",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "required": ["file", "target_lang"],
                                    "properties": {
                                        "file": {
                                            "type": "string",
                                            "format": "binary",
                                            "description": "PDF file to translate",
                                        },
                                        "target_lang": {
                                            "type": "string",
                                            "example": "Tamil",
                                            "description": "Target language name",
                                        },
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "202": {
                            "description": "Job accepted",
                            "content": {"application/json": {"schema": {
                                "type": "object",
                                "properties": {
                                    "job_id":       {"type": "string"},
                                    "status":       {"type": "string", "example": "queued"},
                                    "poll_url":     {"type": "string", "format": "uri"},
                                    "download_url": {"type": "string", "format": "uri"},
                                },
                            }}},
                        },
                        "400": {"description": "Validation error"},
                        "503": {"description": "Server busy — too many active jobs"},
                    },
                }
            },
            "/api/v1/jobs": {
                "get": {
                    "tags": ["jobs"],
                    "summary": "List all jobs",
                    "operationId": "listJobs",
                    "responses": {
                        "200": {"description": "Job list", "content": {"application/json": {"schema": {
                            "type": "object",
                            "properties": {
                                "count": {"type": "integer"},
                                "jobs":  {"type": "array", "items": {"$ref": "#/components/schemas/JobSummary"}},
                            },
                        }}}},
                    },
                }
            },
            "/api/v1/jobs/{job_id}": {
                "get": {
                    "tags": ["jobs"],
                    "summary": "Get job status",
                    "operationId": "getJob",
                    "parameters": [{"name": "job_id", "in": "path", "required": True,
                                    "schema": {"type": "string"}}],
                    "responses": {
                        "200": {"description": "Job detail", "content": {"application/json": {"schema": {
                            "$ref": "#/components/schemas/JobDetail"
                        }}}},
                        "404": {"description": "Job not found"},
                    },
                },
                "delete": {
                    "tags": ["jobs"],
                    "summary": "Delete a completed or failed job",
                    "operationId": "deleteJob",
                    "parameters": [{"name": "job_id", "in": "path", "required": True,
                                    "schema": {"type": "string"}}],
                    "responses": {
                        "200": {"description": "Job deleted"},
                        "404": {"description": "Job not found"},
                        "409": {"description": "Cannot delete an active job"},
                    },
                },
            },
            "/api/v1/jobs/{job_id}/download": {
                "get": {
                    "tags": ["jobs"],
                    "summary": "Download translated PDF",
                    "operationId": "downloadJob",
                    "parameters": [{"name": "job_id", "in": "path", "required": True,
                                    "schema": {"type": "string"}}],
                    "responses": {
                        "200": {"description": "Translated PDF",
                                "content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}}},
                        "202": {"description": "Job still processing"},
                        "404": {"description": "Job not found or file expired"},
                        "500": {"description": "Job failed"},
                    },
                }
            },
        },
        "components": {
            "schemas": {
                "JobSummary": {
                    "type": "object",
                    "properties": {
                        "job_id":       {"type": "string"},
                        "status":       {"type": "string", "enum": ["queued", "processing", "done", "failed"]},
                        "provider":     {"type": "string"},
                        "target_lang":  {"type": "string"},
                        "created_at":   {"type": "number"},
                        "finished_at":  {"type": "number", "nullable": True},
                        "download_url": {"type": "string", "nullable": True},
                    },
                },
                "JobDetail": {
                    "type": "object",
                    "properties": {
                        "job_id":           {"type": "string"},
                        "status":           {"type": "string", "enum": ["queued", "processing", "done", "failed"]},
                        "provider":         {"type": "string"},
                        "model":            {"type": "string"},
                        "target_lang":      {"type": "string"},
                        "filename":         {"type": "string"},
                        "created_at":       {"type": "number"},
                        "started_at":       {"type": "number", "nullable": True},
                        "finished_at":      {"type": "number", "nullable": True},
                        "duration_seconds": {"type": "number", "nullable": True},
                        "error":            {"type": "string", "nullable": True},
                        "result": {
                            "type": "object",
                            "nullable": True,
                            "properties": {
                                "pages":       {"type": "integer"},
                                "spans":       {"type": "integer"},
                                "source_lang": {"type": "string"},
                                "size_mb":     {"type": "number"},
                                "warning":     {"type": "string", "nullable": True},
                            },
                        },
                        "download_url": {"type": "string", "nullable": True},
                    },
                },
            }
        },
    }
    return jsonify(spec)


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

    dispatched = False
    if _celery_task is not None:
        try:
            _celery_task.delay(job_id, str(input_path), str(output_path), params)
            dispatched = True
        except Exception as exc:
            log.warning("Celery dispatch failed (%s) — falling back to thread", exc)
    if not dispatched:
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
