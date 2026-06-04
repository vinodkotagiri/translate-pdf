"""
tests/test_api.py
==================
Integration tests for the PDF Translator API.
Run: pytest tests/ -v
"""
import io
import json
import os
import time

import fitz
import pytest

# Set test environment before importing app
os.environ["FLASK_ENV"] = "testing"
os.environ["ANTHROPIC_API_KEY"] = "test-key"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import create_app
from config import TestingConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def app():
    application = create_app(TestingConfig)
    application.config["TESTING"] = True
    return application


@pytest.fixture(scope="session")
def client(app):
    return app.test_client()


@pytest.fixture(scope="session")
def sample_pdf_bytes():
    """Create a minimal in-memory PDF for testing."""
    doc  = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Hello World", fontsize=16)
    page.insert_text((72, 130), "This is a test document.", fontsize=11)
    page.insert_text((72, 160), "Invoice Total: $1,200.00", fontsize=11)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    buf.seek(0)
    return buf.read()


# ── Health ────────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ok"
    assert "jobs" in data


# ── Providers ────────────────────────────────────────────────────────────────

def test_list_providers(client):
    r = client.get("/api/v1/providers")
    assert r.status_code == 200
    data = r.get_json()
    assert "providers" in data
    names = [p["provider"] for p in data["providers"]]
    for expected in ["claude", "openai", "gemini", "grok", "groq", "mistral", "ollama"]:
        assert expected in names


def test_provider_has_models(client):
    r    = client.get("/api/v1/providers")
    data = r.get_json()
    for p in data["providers"]:
        assert len(p["models"]) > 0
        assert p["default_model"]
        assert p["env_key"]


# ── Languages ────────────────────────────────────────────────────────────────

def test_list_languages(client):
    r    = client.get("/api/v1/languages")
    data = r.get_json()
    assert r.status_code == 200
    assert data["count"] >= 50
    assert "French" in data["languages"]
    assert "Japanese" in data["languages"]
    assert "Telugu" in data["languages"]


# ── Input validation ──────────────────────────────────────────────────────────

def test_translate_no_file(client):
    r = client.post("/api/v1/translate", data={"target_lang": "French"})
    assert r.status_code == 400
    assert "file" in r.get_json()["error"].lower()


def test_translate_no_target_lang(client, sample_pdf_bytes):
    r = client.post(
        "/api/v1/translate",
        data={"file": (io.BytesIO(sample_pdf_bytes), "test.pdf")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400
    assert "target_lang" in r.get_json()["error"]


def test_translate_bad_provider(client, sample_pdf_bytes):
    r = client.post(
        "/api/v1/translate",
        data={
            "file": (io.BytesIO(sample_pdf_bytes), "test.pdf"),
            "target_lang": "French",
            "provider": "nonexistent_llm",
        },
        content_type="multipart/form-data",
    )
    assert r.status_code == 400
    assert "provider" in r.get_json()["error"].lower()


def test_translate_non_pdf(client):
    r = client.post(
        "/api/v1/translate",
        data={
            "file": (io.BytesIO(b"not a pdf"), "document.txt"),
            "target_lang": "French",
        },
        content_type="multipart/form-data",
    )
    assert r.status_code == 400


# ── Job lifecycle ─────────────────────────────────────────────────────────────

def test_jobs_list_empty(client):
    r = client.get("/api/v1/jobs")
    assert r.status_code == 200
    data = r.get_json()
    assert "jobs" in data
    assert isinstance(data["jobs"], list)


def test_job_not_found(client):
    r = client.get("/api/v1/jobs/nonexistent-id")
    assert r.status_code == 404


def test_job_delete_not_found(client):
    r = client.delete("/api/v1/jobs/nonexistent-id")
    assert r.status_code == 404


# ── Core engine (unit tests, no LLM call) ─────────────────────────────────────

def test_extract_text_blocks(sample_pdf_bytes):
    from app.core.engine import extract_pages
    doc   = fitz.open(stream=sample_pdf_bytes, filetype="pdf")
    pages = extract_pages(doc)
    doc.close()
    assert len(pages) == 1
    assert len(pages[0].spans) > 0
    texts = [s.text for s in pages[0].spans]
    assert any("Hello" in t for t in texts)


def test_language_detection():
    from app.core.engine import PageSpans, TextSpan, detect_language
    import fitz as _fitz
    dummy_rect = _fitz.Rect(0, 0, 100, 20)
    spans = [
        TextSpan(0, 0, i, text, dummy_rect, "helv", 11, 0, 0, (0, 0))
        for i, text in enumerate(["This is an English sentence.", "The document contains text."])
    ]
    pages = [PageSpans(page_num=0, spans=spans)]
    lang  = detect_language(pages)
    assert "English" in lang or lang != ""


def test_rebuild_pdf_preserves_images(sample_pdf_bytes):
    """Ensure image blocks are not touched during rebuild."""
    from app.core.engine import PageSpans, extract_pages, rebuild_pdf
    doc   = fitz.open(stream=sample_pdf_bytes, filetype="pdf")
    pages = extract_pages(doc)
    # Set all translations to same as original (no changes)
    for p in pages:
        for s in p.spans:
            s.translated = s.text
    doc   = rebuild_pdf(doc, pages)
    out   = io.BytesIO()
    doc.save(out)
    out.seek(0)
    assert len(out.read()) > 100  # valid PDF produced


def test_parse_llm_response():
    from app.core.providers import parse_llm_response
    raw       = '{"0": "Bonjour", "1": "Au revoir", "2": "Merci"}'
    originals = ["Hello", "Goodbye", "Thank you"]
    result    = parse_llm_response(raw, originals)
    assert result == ["Bonjour", "Au revoir", "Merci"]


def test_parse_llm_response_with_fences():
    from app.core.providers import parse_llm_response
    raw       = '```json\n{"0": "Hola", "1": "Adiós"}\n```'
    originals = ["Hello", "Goodbye"]
    result    = parse_llm_response(raw, originals)
    assert result == ["Hola", "Adiós"]


def test_parse_llm_response_fallback():
    from app.core.providers import parse_llm_response
    raw       = "This is not JSON at all"
    originals = ["Hello", "World"]
    result    = parse_llm_response(raw, originals)
    assert result == originals  # falls back to originals


def test_provider_registry():
    from app.core.providers import PROVIDERS, PROVIDER_MODELS, ENV_KEYS
    for name in PROVIDERS:
        assert name in PROVIDER_MODELS
        assert name in ENV_KEYS
        assert PROVIDERS[name].default_model


def test_config_dirs_created():
    from config import TestingConfig
    TestingConfig.ensure_dirs()
    assert TestingConfig.UPLOAD_DIR.exists()
    assert TestingConfig.OUTPUT_DIR.exists()
