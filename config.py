"""
config.py — Centralised configuration with environment validation.
All settings read from environment variables with sensible defaults.
"""
from __future__ import annotations
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    # ── Flask ────────────────────────────────────────────────────────
    SECRET_KEY        = os.environ.get("SECRET_KEY", os.urandom(32).hex())
    DEBUG             = False
    TESTING           = False
    JSON_SORT_KEYS    = False

    # ── Upload limits ────────────────────────────────────────────────
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_UPLOAD_MB", 100)) * 1024 * 1024
    UPLOAD_DIR         = Path(os.environ.get("UPLOAD_DIR", BASE_DIR / "uploads"))
    OUTPUT_DIR         = Path(os.environ.get("OUTPUT_DIR", BASE_DIR / "outputs"))

    # ── Job lifecycle ────────────────────────────────────────────────
    JOB_TTL_SECONDS    = int(os.environ.get("JOB_TTL_HOURS", 4)) * 3600
    MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", 20))

    # ── Translation engine ───────────────────────────────────────────
    DEFAULT_PROVIDER   = os.environ.get("DEFAULT_PROVIDER", "claude")
    MAX_WORKERS        = int(os.environ.get("MAX_WORKERS", 6))
    CHUNK_SIZE         = int(os.environ.get("CHUNK_SIZE", 20))

    # ── Redis (for Celery + rate limiting) ───────────────────────────
    REDIS_URL          = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL  = os.environ.get("CELERY_BROKER_URL", REDIS_URL)
    CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", REDIS_URL)

    # ── Rate limiting ────────────────────────────────────────────────
    RATELIMIT_ENABLED  = os.environ.get("RATELIMIT_ENABLED", "true").lower() == "true"
    RATELIMIT_DEFAULT  = os.environ.get("RATELIMIT_DEFAULT", "60/hour")
    RATELIMIT_TRANSLATE = os.environ.get("RATELIMIT_TRANSLATE", "20/hour")

    # ── LLM API Keys ─────────────────────────────────────────────────
    ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY     = os.environ.get("OPENAI_API_KEY", "")
    GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")
    GROK_API_KEY       = os.environ.get("GROK_API_KEY", "")
    GROQ_API_KEY       = os.environ.get("GROQ_API_KEY", "")
    MISTRAL_API_KEY    = os.environ.get("MISTRAL_API_KEY", "")
    COHERE_API_KEY     = os.environ.get("COHERE_API_KEY", "")
    OLLAMA_BASE_URL    = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    # ── CORS ─────────────────────────────────────────────────────────
    CORS_ORIGINS       = os.environ.get("CORS_ORIGINS", "*")

    # ── Logging ──────────────────────────────────────────────────────
    LOG_LEVEL          = os.environ.get("LOG_LEVEL", "INFO")
    LOG_DIR            = Path(os.environ.get("LOG_DIR", BASE_DIR / "logs"))

    @classmethod
    def ensure_dirs(cls):
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)


class DevelopmentConfig(Config):
    DEBUG     = True
    LOG_LEVEL = "DEBUG"
    RATELIMIT_ENABLED = False


class ProductionConfig(Config):
    DEBUG     = False
    LOG_LEVEL = "INFO"


class TestingConfig(Config):
    TESTING   = True
    DEBUG     = True
    RATELIMIT_ENABLED = False
    UPLOAD_DIR = Path("/tmp/pdf_translator_test_uploads")
    OUTPUT_DIR = Path("/tmp/pdf_translator_test_outputs")


_configs = {
    "development": DevelopmentConfig,
    "production":  ProductionConfig,
    "testing":     TestingConfig,
}


def get_config() -> type[Config]:
    env = os.environ.get("FLASK_ENV", "production")
    return _configs.get(env, ProductionConfig)
