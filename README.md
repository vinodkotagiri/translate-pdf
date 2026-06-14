# PDF Translator API

Translate any PDF to any language while perfectly preserving layout, images, font sizes, colors, and text positions.

**Two fields. That's it.** Upload your PDF, pick a language — the API handles the rest.

---

## Quick Start

### Synchronous (small documents — up to ~20 pages)

```bash
curl -X POST https://your-domain.com/api/v1/translate \
  -F "file=@invoice.pdf" \
  -F "target_lang=Hindi" \
  --output invoice_hindi.pdf
```

### Asynchronous (large documents — up to 500 pages)

```bash
# 1. Submit — returns a job_id immediately
curl -X POST https://your-domain.com/api/v1/translate/async \
  -F "file=@report.pdf" \
  -F "target_lang=Tamil"
# → {"job_id":"abc123","status":"queued","poll_url":"...","download_url":"..."}

# 2. Poll until done
curl https://your-domain.com/api/v1/jobs/abc123
# → {"status":"done", ...}

# 3. Download
curl https://your-domain.com/api/v1/jobs/abc123/download --output report_tamil.pdf
```

---

## Supported Languages

**Indian languages** (all 22 scheduled):
Hindi, Bengali, Telugu, Marathi, Tamil, Urdu, Gujarati, Kannada, Malayalam, Odia, Punjabi, Assamese, Maithili, Sanskrit, Kashmiri, Sindhi, Konkani, Dogri, Bodo, Nepali, and more.

**World languages**: Arabic, Japanese, Chinese (Simplified/Traditional), Korean, French, German, Spanish, Portuguese, Russian, Italian, and 30+ more.

Full list: `GET /api/v1/languages`

---

## API Reference

### `POST /api/v1/translate` — Synchronous

| Field | Required | Description |
| --- | --- | --- |
| `file` | ✅ | PDF to translate |
| `target_lang` | ✅ | Target language, e.g. `Hindi`, `Tamil`, `French` |

**Response**: PDF file download with metadata headers (`X-Pages`, `X-Spans`, `X-Source-Lang`, `X-Size-MB`).

---

### `POST /api/v1/translate/async` — Asynchronous

Same two fields: `file` and `target_lang`.

**Response** (`202 Accepted`):

```json
{
  "job_id":       "abc123...",
  "status":       "queued",
  "poll_url":     "https://your-domain.com/api/v1/jobs/abc123",
  "download_url": "https://your-domain.com/api/v1/jobs/abc123/download"
}
```

**Job status values**: `queued` → `processing` → `done` | `failed`

---

### `GET /api/v1/jobs/<id>` — Poll job status

```json
{
  "job_id":           "abc123",
  "status":           "done",
  "target_lang":      "Hindi",
  "duration_seconds": 42.1,
  "result": {
    "pages":       12,
    "spans":       384,
    "source_lang": "English",
    "size_mb":     1.2
  },
  "download_url": "https://your-domain.com/api/v1/jobs/abc123/download"
}
```

### `GET /api/v1/jobs/<id>/download` — Download translated PDF

Returns the translated PDF (only when `status == "done"`).

### `DELETE /api/v1/jobs/<id>` — Delete a job

Removes the job and its files.

### `GET /api/v1/jobs` — List all jobs

### `GET /health` — Health check

### `GET /api/v1/languages` — List all supported languages

---

## Interactive Docs (Swagger UI)

Open **`/api/docs`** in your browser to explore and test all endpoints interactively.

---

## Limits

| Limit | Value |
| --- | --- |
| Max file size | 100 MB |
| Max pages | 500 |
| Rate limit | 20 requests / hour |

---

## Deploy

```bash
git clone <your-repo>
cd pdf_translator_prod

cp .env.example .env
# Set OPENAI_API_KEY in .env

docker compose up -d --build

curl http://localhost:5000/health
```

---

## Troubleshooting

**Scanned/image-only PDFs return empty output:**
Run OCR first — `ocrmypdf input.pdf input_ocr.pdf` — then translate.

**Encrypted PDFs:**
Decrypt first — `qpdf --decrypt --password=yourpassword encrypted.pdf decrypted.pdf`

**Timeout on large documents:**
Use the async endpoint (`/api/v1/translate/async`) for anything over 20 pages.
