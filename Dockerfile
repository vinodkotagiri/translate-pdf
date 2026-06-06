# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps for PyMuPDF + Unicode fonts for Indian script rendering.
# fonts-indic      → meta-package that installs Lohit fonts for all 22 Indian
#                    scheduled languages (Bengali, Tamil, Telugu, Kannada,
#                    Malayalam, Gujarati, Gurmukhi, Odia, Devanagari …)
# fonts-noto-core  → NotoSans fallback fonts + Arabic, Hebrew, Thai …
# fonts-noto-cjk   → Japanese, Chinese, Korean
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libmupdf-dev \
        libfreetype6 \
        libharfbuzz0b \
        libjpeg62-turbo \
        libopenjp2-7 \
        fonts-indic \
        fonts-noto-core \
        fonts-noto-extra \
        fonts-noto-cjk \
    && (apt-get install -y --no-install-recommends \
        fonts-deva fonts-beng fonts-taml fonts-telu \
        fonts-mlym fonts-gujr fonts-knda fonts-orya fonts-guru \
        || echo "Note: some script-specific font packages not found on this Debian version") \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application
COPY --chown=appuser:appuser . .

# Create runtime dirs
RUN mkdir -p uploads outputs logs \
    && chown -R appuser:appuser /app

USER appuser

# Environment
ENV FLASK_ENV=production \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

CMD ["gunicorn", "wsgi:application", "-c", "deploy/gunicorn.conf.py"]
