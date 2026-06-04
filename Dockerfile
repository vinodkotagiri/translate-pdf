# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps for PyMuPDF + Unicode fonts for non-Latin script rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmupdf-dev \
        libfreetype6 \
        libharfbuzz0b \
        libjpeg62-turbo \
        libopenjp2-7 \
        fonts-noto-core \
        fonts-noto-cjk \
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
