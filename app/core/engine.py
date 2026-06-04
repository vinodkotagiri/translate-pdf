"""
app/core/engine.py
==================
Production-grade PDF translation engine.
Extracts text spans with full styling → translates in parallel → reconstructs PDF.
All images, positions, colors, and font metrics are preserved.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from app.core.providers import LLMProvider, get_provider

log = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 2
REDACT_COLOR    = (1.0, 1.0, 1.0)  # white

# ── Unicode font paths (Noto fonts installed in Docker image) ──────────────────

_NOTO_TTF = Path("/usr/share/fonts/truetype/noto")
_NOTO_OTF = Path("/usr/share/fonts/opentype/noto")

# Maps target language name (as returned by LANG_CODE_MAP) → candidate font files.
# First existing file wins.
_LANG_FONT_CANDIDATES: dict[str, list[str]] = {
    "Hindi":                 [str(_NOTO_TTF / "NotoSansDevanagari-Regular.ttf")],
    "Marathi":               [str(_NOTO_TTF / "NotoSansDevanagari-Regular.ttf")],
    "Nepali":                [str(_NOTO_TTF / "NotoSansDevanagari-Regular.ttf")],
    "Bengali":               [str(_NOTO_TTF / "NotoSansBengali-Regular.ttf")],
    "Tamil":                 [str(_NOTO_TTF / "NotoSansTamil-Regular.ttf")],
    "Telugu":                [str(_NOTO_TTF / "NotoSansTelugu-Regular.ttf")],
    "Kannada":               [str(_NOTO_TTF / "NotoSansKannada-Regular.ttf")],
    "Malayalam":             [str(_NOTO_TTF / "NotoSansMalayalam-Regular.ttf")],
    "Gujarati":              [str(_NOTO_TTF / "NotoSansGujarati-Regular.ttf")],
    "Punjabi":               [str(_NOTO_TTF / "NotoSansGurmukhi-Regular.ttf")],
    "Arabic":                [str(_NOTO_TTF / "NotoNaskhArabic-Regular.ttf")],
    "Urdu":                  [str(_NOTO_TTF / "NotoNaskhArabic-Regular.ttf")],
    "Persian":               [str(_NOTO_TTF / "NotoNaskhArabic-Regular.ttf")],
    "Hebrew":                [str(_NOTO_TTF / "NotoSansHebrew-Regular.ttf")],
    "Thai":                  [str(_NOTO_TTF / "NotoSansThai-Regular.ttf")],
    "Japanese":              [str(_NOTO_OTF / "NotoSansCJK-Regular.ttc")],
    "Chinese (Simplified)":  [str(_NOTO_OTF / "NotoSansCJK-Regular.ttc")],
    "Chinese (Traditional)": [str(_NOTO_OTF / "NotoSansCJK-Regular.ttc")],
    "Korean":                [str(_NOTO_OTF / "NotoSansCJK-Regular.ttc")],
    "Greek":                 [str(_NOTO_TTF / "NotoSans-Regular.ttf")],
    "Russian":               [str(_NOTO_TTF / "NotoSans-Regular.ttf")],
    "Ukrainian":             [str(_NOTO_TTF / "NotoSans-Regular.ttf")],
    "Bulgarian":             [str(_NOTO_TTF / "NotoSans-Regular.ttf")],
    "Serbian":               [str(_NOTO_TTF / "NotoSans-Regular.ttf")],
    "Macedonian":            [str(_NOTO_TTF / "NotoSans-Regular.ttf")],
}

_font_cache: dict[str, str | None] = {}


def _get_unicode_font(target_lang: str) -> str | None:
    """Return path to a Noto font file for the target language, or None to use built-in Latin fonts."""
    if target_lang not in _font_cache:
        candidates = _LANG_FONT_CANDIDATES.get(target_lang, [])
        _font_cache[target_lang] = next(
            (p for p in candidates if Path(p).exists()), None
        )
    return _font_cache[target_lang]


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class TextSpan:
    page_num:   int
    block_idx:  int
    span_idx:   int
    text:       str
    rect:       fitz.Rect
    font_name:  str
    font_size:  float
    font_flags: int
    color:      int          # packed 0xRRGGBB
    origin:     tuple[float, float]
    translated: str = ""


@dataclass
class PageSpans:
    page_num: int
    spans:    list[TextSpan] = field(default_factory=list)


# ── Extraction ─────────────────────────────────────────────────────────────────

def extract_pages(doc: fitz.Document) -> list[PageSpans]:
    """
    Extract all text spans from every page using PyMuPDF rawdict.
    Compatible with PyMuPDF ≥1.18 (handles both 'text' and 'chars' span formats).
    """
    result: list[PageSpans] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_spans = PageSpans(page_num=page_num)
        raw = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for b_idx, block in enumerate(raw.get("blocks", [])):
            if block.get("type") != 0:
                continue  # skip image blocks
            for line in block.get("lines", []):
                for s_idx, span in enumerate(line.get("spans", [])):
                    # Handle both PyMuPDF span formats
                    if "text" in span:
                        text = span["text"].strip()
                    elif "chars" in span:
                        text = "".join(c.get("c", "") for c in span["chars"]).strip()
                    else:
                        continue

                    if len(text) < MIN_TEXT_LENGTH:
                        continue

                    rect = fitz.Rect(span["bbox"])
                    page_spans.spans.append(TextSpan(
                        page_num   = page_num,
                        block_idx  = b_idx,
                        span_idx   = s_idx,
                        text       = text,
                        rect       = rect,
                        font_name  = span.get("font", "helv"),
                        font_size  = span.get("size", 11.0),
                        font_flags = span.get("flags", 0),
                        color      = span.get("color", 0),
                        origin     = tuple(span.get("origin", (rect.x0, rect.y1))),
                    ))

        result.append(page_spans)

    total = sum(len(p.spans) for p in result)
    log.info(f"Extracted {total} text spans from {len(result)} pages.")
    return result


# ── Language detection ──────────────────────────────────────────────────────────

LANG_CODE_MAP = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali",
    "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "et": "Estonian", "fa": "Persian",
    "fi": "Finnish", "fr": "French", "gu": "Gujarati", "he": "Hebrew",
    "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian", "id": "Indonesian",
    "it": "Italian", "ja": "Japanese", "kn": "Kannada", "ko": "Korean",
    "lt": "Lithuanian", "lv": "Latvian", "mk": "Macedonian", "ml": "Malayalam",
    "mr": "Marathi", "ms": "Malay", "mt": "Maltese", "nl": "Dutch",
    "no": "Norwegian", "pl": "Polish", "pt": "Portuguese", "pa": "Punjabi",
    "ro": "Romanian", "ru": "Russian", "sk": "Slovak", "sl": "Slovenian",
    "sq": "Albanian", "sr": "Serbian", "sv": "Swedish", "sw": "Swahili",
    "ta": "Tamil", "te": "Telugu", "th": "Thai", "tr": "Turkish",
    "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese", "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)", "cy": "Welsh",
}


def detect_language(pages: list[PageSpans]) -> str:
    sample = " ".join(
        s.text for p in pages[:3] for s in p.spans[:20]
    ).strip()
    if not sample:
        return "the source language"
    try:
        from langdetect import detect
        code = detect(sample)
        return LANG_CODE_MAP.get(code, code.upper())
    except Exception:
        return "the source language"


# ── Translation ─────────────────────────────────────────────────────────────────

def _translate_page(
    page:        PageSpans,
    target_lang: str,
    source_lang: str,
    provider:    LLMProvider,
    chunk_size:  int,
) -> PageSpans:
    if not page.spans:
        return page
    texts = [s.text for s in page.spans]
    translated: list[str] = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i: i + chunk_size]
        result = provider.translate_batch(chunk, target_lang, source_lang)
        translated.extend(result)
    for span, t in zip(page.spans, translated):
        span.translated = t
    return page


def translate_pages(
    pages:       list[PageSpans],
    target_lang: str,
    source_lang: str,
    provider:    LLMProvider,
    max_workers: int,
    chunk_size:  int,
    progress_cb  = None,  # optional callable(completed, total)
) -> list[PageSpans]:
    """Translate all pages in parallel. Calls progress_cb after each page."""
    results = [None] * len(pages)
    total   = len(pages)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_translate_page, p, target_lang, source_lang, provider, chunk_size): i
            for i, p in enumerate(pages)
        }
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                log.error(f"Page {idx} translation failed: {exc}")
                results[idx] = pages[idx]  # keep original on failure
            done += 1
            if progress_cb:
                try:
                    progress_cb(done, total)
                except Exception:
                    pass

    return results


# ── PDF reconstruction ──────────────────────────────────────────────────────────

def _color_from_int(packed: int) -> tuple[float, float, float]:
    return (
        ((packed >> 16) & 0xFF) / 255.0,
        ((packed >>  8) & 0xFF) / 255.0,
        ((packed)       & 0xFF) / 255.0,
    )


def _resolve_font(font_name: str, flags: int) -> str:
    """Map original font to a built-in PyMuPDF font that renders reliably."""
    n = font_name.lower()
    if any(k in n for k in ("mono", "courier", "cour", "consol", "code", "fixed")):
        return "cour"
    if any(k in n for k in ("times", "serif", "georgia", "garamond")):
        return "tiro"
    return "helv"


def _fit_size(
    text: str,
    font: str,
    orig_size: float,
    rect: fitz.Rect,
    min_size: float = 4.0,
    font_file: str | None = None,
) -> float:
    """Shrink font size until the text fits within the span's width."""
    size = orig_size
    while size >= min_size:
        try:
            if font_file:
                tw = fitz.get_text_length(text, fontfile=font_file, fontname=font, fontsize=size)
            else:
                tw = fitz.get_text_length(text, fontname=font, fontsize=size)
        except Exception:
            tw = len(text) * size * 0.55
        if tw <= rect.width * 1.15:
            break
        size -= 0.5
    return max(size, min_size)


def rebuild_pdf(doc: fitz.Document, pages: list[PageSpans], target_lang: str = "") -> fitz.Document:
    """
    For each translated span:
      1. Erase original text with a white filled rectangle
      2. Insert translated text at the exact baseline with matching style

    For non-Latin target languages (Hindi, Arabic, CJK, etc.) a Noto font file
    is used so glyphs are actually present; built-in PDF fonts only cover Latin.
    """
    unicode_font = _get_unicode_font(target_lang) if target_lang else None
    if target_lang and not unicode_font:
        log.info("No unicode font found for '%s', falling back to built-in Latin fonts.", target_lang)

    for page_data in pages:
        page = doc[page_data.page_num]
        for span in page_data.spans:
            if not span.translated or span.translated == span.text:
                continue

            color = _color_from_int(span.color)

            if unicode_font:
                font_alias = "F0"
                size = _fit_size(
                    span.translated, font_alias, span.font_size, span.rect,
                    font_file=unicode_font,
                )
            else:
                font_alias = _resolve_font(span.font_name, span.font_flags)
                size = _fit_size(span.translated, font_alias, span.font_size, span.rect)

            # Erase original
            erase_rect = fitz.Rect(
                span.rect.x0 - 1, span.rect.y0 - 1,
                span.rect.x1 + 1, span.rect.y1 + 1,
            )
            shape = page.new_shape()
            shape.draw_rect(erase_rect)
            shape.finish(color=None, fill=REDACT_COLOR, width=0)
            shape.commit()

            # Insert translation
            pt = fitz.Point(span.origin[0], span.origin[1])
            try:
                if unicode_font:
                    page.insert_text(
                        pt, span.translated,
                        fontfile=unicode_font, fontname=font_alias,
                        fontsize=size, color=color,
                    )
                else:
                    page.insert_text(pt, span.translated, fontname=font_alias, fontsize=size, color=color)
            except Exception:
                try:
                    page.insert_text(pt, span.translated, fontname="helv", fontsize=size, color=color)
                except Exception as exc:
                    log.warning("Insert failed p%d: %s", page_data.page_num, exc)

    return doc


# ── Main public entry point ─────────────────────────────────────────────────────

def translate_pdf_file(
    input_path:    str | Path,
    output_path:   str | Path,
    target_lang:   str,
    provider_name: str           = "claude",
    api_key:       Optional[str] = None,
    model:         Optional[str] = None,
    max_workers:   int           = 6,
    chunk_size:    int           = 20,
    source_lang:   Optional[str] = None,
    config                       = None,
    progress_cb                  = None,
    **provider_kwargs,
) -> dict:
    """
    Full translate-PDF pipeline.

    Returns a metadata dict:
      pages, spans, provider, model, source_lang, target_lang, size_mb, output_path
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_path}")
    if not input_path.suffix.lower() == ".pdf":
        raise ValueError(f"Input must be a PDF file, got: {input_path.suffix}")

    # Init provider
    provider = get_provider(
        provider_name, api_key=api_key, model=model, config=config, **provider_kwargs
    )
    log.info(f"Provider: {provider}")

    doc     = fitz.open(str(input_path))
    n_pages = len(doc)
    log.info(f"Opened PDF: {input_path.name} ({n_pages} pages)")

    pages      = extract_pages(doc)
    total_spans = sum(len(p.spans) for p in pages)

    if total_spans == 0:
        log.warning("No text spans found — PDF may be image-only.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path))
        doc.close()
        return {
            "output_path": str(output_path),
            "pages": n_pages, "spans": 0,
            "provider": provider_name, "model": provider.model,
            "source_lang": "unknown", "target_lang": target_lang,
            "warning": "No translatable text found (scanned/image-only PDF?)",
        }

    src_lang = source_lang or detect_language(pages)
    log.info(f"Language: {src_lang} → {target_lang}  |  {total_spans} spans")

    pages = translate_pages(
        pages, target_lang, src_lang, provider,
        max_workers=max_workers, chunk_size=chunk_size,
        progress_cb=progress_cb,
    )

    doc = rebuild_pdf(doc, pages, target_lang=target_lang)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path), garbage=4, deflate=True, clean=True)
    doc.close()

    size_mb = output_path.stat().st_size / 1_048_576
    log.info(f"✓ Saved: {output_path.name} ({size_mb:.2f} MB)")

    return {
        "output_path": str(output_path),
        "pages":       n_pages,
        "spans":       total_spans,
        "provider":    provider_name,
        "model":       provider.model,
        "source_lang": src_lang,
        "target_lang": target_lang,
        "size_mb":     round(size_mb, 3),
    }
