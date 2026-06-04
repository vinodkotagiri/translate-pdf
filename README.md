# PDF Translator — Production Server

Multi-LLM PDF translation service. Translates any PDF to any language while perfectly preserving images, layout, text positions, font sizes, colors, and weights.

## File Structure

```
pdf_translator_prod/
├── app/
│   ├── __init__.py           # Flask app factory
│   ├── api/
│   │   └── routes.py         # All REST endpoints
│   └── core/
│       ├── engine.py         # PDF extract → translate → rebuild
│       ├── providers.py      # 8 LLM provider implementations
│       └── job_store.py      # Redis + in-memory job store
├── deploy/
│   ├── gunicorn.conf.py      # Gunicorn production config
│   └── nginx.conf            # Nginx reverse proxy
├── tests/
│   └── test_api.py           # 19 integration + unit tests
├── scripts/
│   └── server.sh             # Start / stop / logs management
├── config.py                 # All configuration
├── wsgi.py                   # WSGI entrypoint
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Quick Deploy

### Option 1 — Docker Compose (recommended)

```bash
git clone <your-repo>
cd pdf_translator_prod

cp .env.example .env
# Edit .env — add your LLM API key(s)

docker compose up -d --build

# Check it's running
curl http://localhost:5000/health
```

### Option 2 — Direct on server (Ubuntu/Debian)

```bash
# 1. Install Python 3.11+
sudo apt-get install python3.11 python3.11-venv redis-server -y

# 2. Create virtualenv
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
nano .env   # Add your API keys

# 4. Run tests
python -m pytest tests/ -v

# 5. Start
bash scripts/server.sh start

# View logs
bash scripts/server.sh logs
```

### Option 3 — Bare Gunicorn (fastest for dev/staging)

```bash
source venv/bin/activate
cp .env.example .env && nano .env

gunicorn wsgi:application \
  -c deploy/gunicorn.conf.py \
  --bind 0.0.0.0:5000
```

---

## Supported LLM Providers

| Provider | .env key | Default model |
|---|---|---|
| `claude` | `ANTHROPIC_API_KEY` | claude-opus-4-5 |
| `openai` | `OPENAI_API_KEY` | gpt-4o |
| `gemini` | `GEMINI_API_KEY` | gemini-2.0-flash |
| `grok` | `GROK_API_KEY` | grok-3 |
| `groq` | `GROQ_API_KEY` | llama-3.3-70b-versatile |
| `mistral` | `MISTRAL_API_KEY` | mistral-large-latest |
| `cohere` | `COHERE_API_KEY` | command-r-plus |
| `ollama` | *(no key)* | llama3 |

Only set keys for providers you want to use. The others will return 500 if called.

---

## REST API Reference

### `GET /health`
Liveness + readiness probe. Returns job queue counts.

### `GET /api/v1/providers`
Lists all providers, their models, and whether the API key is configured.

### `GET /api/v1/languages`
Lists 58 supported target languages.

---

### `POST /api/v1/translate` — Synchronous

Blocks until translation completes, returns PDF directly.

**Form fields:**
| Field | Required | Description |
|---|---|---|
| `file` | ✅ | PDF binary upload |
| `target_lang` | ✅ | e.g. `French`, `Japanese`, `Telugu` |
| `provider` | – | default from `DEFAULT_PROVIDER` env |
| `model` | – | override default model |
| `api_key` | – | override env-var key |
| `source_lang` | – | skip auto-detection |
| `max_workers` | – | 1–20, default 6 |
| `chunk_size` | – | 1–50, default 20 |

**Response headers:** `X-Pages`, `X-Spans`, `X-Provider`, `X-Model`, `X-Source-Lang`, `X-Target-Lang`, `X-Size-MB`

```bash
# Translate with Claude (default)
curl -X POST https://your-domain.com/api/v1/translate \
  -F "file=@invoice.pdf" \
  -F "target_lang=French" \
  --output invoice_fr.pdf

# With specific provider + model
curl -X POST https://your-domain.com/api/v1/translate \
  -F "file=@manual.pdf" \
  -F "target_lang=Japanese" \
  -F "provider=openai" \
  -F "model=gpt-4o" \
  --output manual_ja.pdf

# Groq (fastest)
curl -X POST https://your-domain.com/api/v1/translate \
  -F "file=@report.pdf" \
  -F "target_lang=German" \
  -F "provider=groq" \
  --output report_de.pdf

# Gemini
curl -X POST https://your-domain.com/api/v1/translate \
  -F "file=@contract.pdf" \
  -F "target_lang=Spanish" \
  -F "provider=gemini" \
  --output contract_es.pdf
```

---

### `POST /api/v1/translate/async` — Asynchronous

Returns immediately with a `job_id`. Use for large documents.

```bash
# Submit
curl -X POST https://your-domain.com/api/v1/translate/async \
  -F "file=@big_document.pdf" \
  -F "target_lang=French" \
  -F "provider=openai"
# → {"job_id":"abc...","status":"queued","poll_url":"...","download_url":"..."}

# Poll (repeat until status == "done")
curl https://your-domain.com/api/v1/jobs/abc...

# Download
curl https://your-domain.com/api/v1/jobs/abc.../download --output result.pdf
```

**Job status values:** `queued` → `processing` → `done` | `failed`

---

### `GET /api/v1/jobs`
List all jobs.

### `GET /api/v1/jobs/<id>`
Full job metadata including duration, spans translated, source language detected.

### `GET /api/v1/jobs/<id>/download`
Download translated PDF (only when `status == "done"`).

### `DELETE /api/v1/jobs/<id>`
Delete job and its temp files.

---

## Configuration Reference (`.env`)

| Variable | Default | Description |
|---|---|---|
| `FLASK_ENV` | `production` | `development` / `production` / `testing` |
| `SECRET_KEY` | *(required)* | Random string — change before deploy |
| `PORT` | `5000` | Server port |
| `DEFAULT_PROVIDER` | `claude` | Fallback provider if none specified |
| `MAX_WORKERS` | `6` | Parallel page translation threads |
| `CHUNK_SIZE` | `20` | Text spans per LLM API call |
| `MAX_UPLOAD_MB` | `100` | Max PDF file size |
| `JOB_TTL_HOURS` | `4` | Auto-expire async jobs after N hours |
| `MAX_CONCURRENT_JOBS` | `20` | Reject requests beyond this |
| `REDIS_URL` | — | Redis URL — enables persistent job store |
| `RATELIMIT_ENABLED` | `true` | Toggle rate limiting |
| `RATELIMIT_TRANSLATE` | `20/hour` | Rate limit on `/translate` endpoints |
| `GUNICORN_WORKERS` | `4` | Gunicorn worker processes |
| `GUNICORN_THREADS` | `4` | Threads per worker |
| `GUNICORN_TIMEOUT` | `600` | Request timeout in seconds |
| `CORS_ORIGINS` | `*` | Allowed origins (comma-separated) |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |

---

## Adding a New LLM Provider

Edit `app/core/providers.py`:

```python
class MyProvider(LLMProvider):
    name          = "myprovider"
    default_model = "my-model-v1"

    def _init_client(self, **kwargs):
        import my_sdk
        self.client = my_sdk.Client(api_key=self.api_key)

    @with_retry
    def translate_batch(self, texts, target_lang, source_lang):
        resp = self.client.complete(build_translation_prompt(texts, target_lang, source_lang))
        return parse_llm_response(resp.text, texts)

# Register
PROVIDERS["myprovider"]       = MyProvider
PROVIDER_MODELS["myprovider"] = ["my-model-v1", "my-model-fast"]
ENV_KEYS["myprovider"]        = "MYPROVIDER_API_KEY"
```

That's it — CLI, API, job store, and rate limiting all pick it up automatically.

---

## Troubleshooting

**Scanned PDFs return 0 spans:**
```bash
# Run OCR first
pip install ocrmypdf
ocrmypdf input.pdf input_ocr.pdf
# Then translate input_ocr.pdf
```

**Encrypted PDFs:**
```bash
qpdf --decrypt --password=yourpassword encrypted.pdf decrypted.pdf
```

**Timeouts on large docs:**
Increase `GUNICORN_TIMEOUT=1200` and use async endpoint for anything > 20 pages.

**Redis connection refused:**
The job store falls back to in-memory automatically. For multi-worker deployments you need Redis — start it with `docker compose up redis -d`.

---

## License
MIT
