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
import unicodedata
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

# Maps target language name → regular and bold Noto font file candidates.
# First existing path wins per weight.
_LANG_FONT_CANDIDATES: dict[str, dict[str, list[str]]] = {
    # Lohit-Devanagari covers Devanagari + Basic Latin + ₹ in one file, so Latin
    # abbreviations (JLR, TML, CV, EBITDA …) render correctly inside a single textbox.
    # NotoSansDevanagari is kept as fallback and for the bold weight.
    "Hindi":                 {"r": ["/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
                                    str(_NOTO_TTF / "NotoSansDevanagari-Regular.ttf")],
                              "b": [str(_NOTO_TTF / "NotoSansDevanagari-Bold.ttf")]},
    "Marathi":               {"r": ["/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
                                    str(_NOTO_TTF / "NotoSansDevanagari-Regular.ttf")],
                              "b": [str(_NOTO_TTF / "NotoSansDevanagari-Bold.ttf")]},
    "Nepali":                {"r": ["/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
                                    str(_NOTO_TTF / "NotoSansDevanagari-Regular.ttf")],
                              "b": [str(_NOTO_TTF / "NotoSansDevanagari-Bold.ttf")]},
    "Bengali":               {"r": [str(_NOTO_TTF / "NotoSansBengali-Regular.ttf")],
                              "b": [str(_NOTO_TTF / "NotoSansBengali-Bold.ttf")]},
    "Tamil":                 {"r": [str(_NOTO_TTF / "NotoSansTamil-Regular.ttf")],
                              "b": [str(_NOTO_TTF / "NotoSansTamil-Bold.ttf")]},
    "Telugu":                {"r": [str(_NOTO_TTF / "NotoSansTelugu-Regular.ttf")],
                              "b": [str(_NOTO_TTF / "NotoSansTelugu-Bold.ttf")]},
    "Kannada":               {"r": [str(_NOTO_TTF / "NotoSansKannada-Regular.ttf")],
                              "b": [str(_NOTO_TTF / "NotoSansKannada-Bold.ttf")]},
    "Malayalam":             {"r": [str(_NOTO_TTF / "NotoSansMalayalam-Regular.ttf")],
                              "b": [str(_NOTO_TTF / "NotoSansMalayalam-Bold.ttf")]},
    "Gujarati":              {"r": [str(_NOTO_TTF / "NotoSansGujarati-Regular.ttf")],
                              "b": [str(_NOTO_TTF / "NotoSansGujarati-Bold.ttf")]},
    "Punjabi":               {"r": [str(_NOTO_TTF / "NotoSansGurmukhi-Regular.ttf")],
                              "b": [str(_NOTO_TTF / "NotoSansGurmukhi-Bold.ttf")]},
    "Arabic":                {"r": [str(_NOTO_TTF / "NotoNaskhArabic-Regular.ttf")],
                              "b": [str(_NOTO_TTF / "NotoNaskhArabic-Bold.ttf")]},
    "Urdu":                  {"r": [str(_NOTO_TTF / "NotoNaskhArabic-Regular.ttf")],
                              "b": [str(_NOTO_TTF / "NotoNaskhArabic-Bold.ttf")]},
    "Persian":               {"r": [str(_NOTO_TTF / "NotoNaskhArabic-Regular.ttf")],
                              "b": [str(_NOTO_TTF / "NotoNaskhArabic-Bold.ttf")]},
    "Hebrew":                {"r": [str(_NOTO_TTF / "NotoSansHebrew-Regular.ttf")],
                              "b": [str(_NOTO_TTF / "NotoSansHebrew-Bold.ttf")]},
    "Thai":                  {"r": [str(_NOTO_TTF / "NotoSansThai-Regular.ttf")],
                              "b": [str(_NOTO_TTF / "NotoSansThai-Bold.ttf")]},
    "Japanese":              {"r": [str(_NOTO_OTF / "NotoSansCJK-Regular.ttc")],
                              "b": [str(_NOTO_OTF / "NotoSansCJK-Bold.ttc")]},
    "Chinese (Simplified)":  {"r": [str(_NOTO_OTF / "NotoSansCJK-Regular.ttc")],
                              "b": [str(_NOTO_OTF / "NotoSansCJK-Bold.ttc")]},
    "Chinese (Traditional)": {"r": [str(_NOTO_OTF / "NotoSansCJK-Regular.ttc")],
                              "b": [str(_NOTO_OTF / "NotoSansCJK-Bold.ttc")]},
    "Korean":                {"r": [str(_NOTO_OTF / "NotoSansCJK-Regular.ttc")],
                              "b": [str(_NOTO_OTF / "NotoSansCJK-Bold.ttc")]},
    "Greek":                 {"r": [str(_NOTO_TTF / "NotoSans-Regular.ttf")],
                              "b": [str(_NOTO_TTF / "NotoSans-Bold.ttf")]},
    "Russian":               {"r": [str(_NOTO_TTF / "NotoSans-Regular.ttf")],
                              "b": [str(_NOTO_TTF / "NotoSans-Bold.ttf")]},
    "Ukrainian":             {"r": [str(_NOTO_TTF / "NotoSans-Regular.ttf")],
                              "b": [str(_NOTO_TTF / "NotoSans-Bold.ttf")]},
    "Bulgarian":             {"r": [str(_NOTO_TTF / "NotoSans-Regular.ttf")],
                              "b": [str(_NOTO_TTF / "NotoSans-Bold.ttf")]},
    "Serbian":               {"r": [str(_NOTO_TTF / "NotoSans-Regular.ttf")],
                              "b": [str(_NOTO_TTF / "NotoSans-Bold.ttf")]},
    "Macedonian":            {"r": [str(_NOTO_TTF / "NotoSans-Regular.ttf")],
                              "b": [str(_NOTO_TTF / "NotoSans-Bold.ttf")]},
}

# (lang, is_bold) → resolved font path or None
_font_cache: dict[tuple[str, bool], str | None] = {}


def _get_unicode_font(target_lang: str, is_bold: bool = False) -> str | None:
    """Return path to a Noto font for the target language and weight, or None."""
    key = (target_lang, is_bold)
    if key not in _font_cache:
        weight_key = "b" if is_bold else "r"
        candidates = _LANG_FONT_CANDIDATES.get(target_lang, {}).get(weight_key, [])
        found = next((p for p in candidates if Path(p).exists()), None)
        if is_bold and not found:
            found = _get_unicode_font(target_lang, is_bold=False)  # fall back to regular
        _font_cache[key] = found
    return _font_cache[key]


# ── Mixed-script text helpers ──────────────────────────────────────────────────

# Unicode ranges whose glyphs are absent from Latin-only PDF built-in fonts.
_NON_LATIN_RANGES = (
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Oriya
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x0590, 0x05FF),  # Hebrew
    (0x0E00, 0x0E7F),  # Thai
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0xAC00, 0xD7AF),  # Hangul
    # Symbols absent from all Base-14 PDF fonts — route to Noto so they render
    (0x20A0, 0x20CF),  # Currency Symbols (₹ U+20B9, € U+20AC, etc.)
    (0x2018, 0x201F),  # Typographic quotes
    (0x2013, 0x2014),  # En/em dashes (some Base-14 versions lack them)
)


def _needs_unicode_font(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _NON_LATIN_RANGES)


def _split_text_runs(text: str) -> list[tuple[bool, str]]:
    """
    Split text into (needs_unicode_font, substring) runs.
    Unicode combining marks (matras, diacritics) stay attached to the preceding run
    so Devanagari conjuncts are never broken.
    """
    if not text:
        return []

    def is_combining(ch: str) -> bool:
        return unicodedata.category(ch) in ('Mn', 'Mc', 'Me', 'Cf')

    first_real = next((c for c in text if not is_combining(c)), text[0])
    cur_unicode = _needs_unicode_font(first_real)
    cur_run = ""
    runs: list[tuple[bool, str]] = []

    for ch in text:
        if is_combining(ch):
            cur_run += ch  # glue to current run regardless of script
            continue
        needs = _needs_unicode_font(ch)
        if needs == cur_unicode:
            cur_run += ch
        else:
            if cur_run:
                runs.append((cur_unicode, cur_run))
            cur_unicode = needs
            cur_run = ch

    if cur_run:
        runs.append((cur_unicode, cur_run))

    return runs


def _dir_to_rotate(direction: tuple[float, float]) -> int:
    """Convert a (cos θ, sin θ) text direction to a PyMuPDF insert_text rotate angle."""
    cos_a, sin_a = direction
    if abs(cos_a) >= 0.9:           # horizontal
        return 0 if cos_a > 0 else 180
    return 90 if sin_a > 0 else 270  # vertical: 90 = bottom-to-top, 270 = top-to-bottom


def _insert_text_mixed(
    page: fitz.Page,
    pt: fitz.Point,
    text: str,
    size: float,
    color: tuple,
    unicode_font: str | None,
    latin_font: str,
    unicode_alias: str = "F0",
    rotate: int = 0,
) -> None:
    """
    Insert text that may mix Devanagari/Arabic/CJK and Latin characters.
    Each script run is rendered with the correct font so no glyphs go missing.
    `unicode_alias` must be unique per distinct font file registered on the page
    (use "F0" for regular weight, "F1" for bold).
    """
    if not unicode_font:
        try:
            page.insert_text(pt, text, fontname=latin_font, fontsize=size, color=color, rotate=rotate)
        except Exception as exc:
            log.warning("Insert failed: %s", exc)
        return

    runs = _split_text_runs(text)

    # All-Latin fast path — skip per-run cursor arithmetic
    if len(runs) == 1 and not runs[0][0]:
        try:
            page.insert_text(pt, text, fontname=latin_font, fontsize=size, color=color, rotate=rotate)
        except Exception as exc:
            log.warning("Insert failed: %s", exc)
        return

    x, y = pt.x, pt.y
    for needs_unicode, run_text in runs:
        if not run_text:
            continue
        tw = 0.0
        if needs_unicode:
            try:
                tw = fitz.get_text_length(
                    run_text, fontfile=unicode_font, fontname=unicode_alias, fontsize=size,
                )
            except Exception:
                tw = len(run_text) * size * 0.6
            try:
                page.insert_text(
                    fitz.Point(x, y), run_text,
                    fontfile=unicode_font, fontname=unicode_alias,
                    fontsize=size, color=color, rotate=rotate,
                )
            except Exception as exc:
                log.warning("Unicode run insert failed: %s", exc)
        else:
            try:
                tw = fitz.get_text_length(run_text, fontname=latin_font, fontsize=size)
            except Exception:
                tw = len(run_text) * size * 0.55
            inserted = False
            try:
                page.insert_text(
                    fitz.Point(x, y), run_text,
                    fontname=latin_font, fontsize=size, color=color, rotate=rotate,
                )
                inserted = True
            except Exception:
                pass
            if not inserted:
                try:
                    page.insert_text(
                        fitz.Point(x, y), run_text,
                        fontname="helv", fontsize=size, color=color, rotate=rotate,
                    )
                except Exception as exc:
                    log.warning("Latin run insert failed: %s", exc)
        # For vertical text, advance along y-axis instead of x
        if rotate in (90, 270):
            y += tw * (-1 if rotate == 90 else 1)
        else:
            x += tw


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
    direction:  tuple[float, float] = (1.0, 0.0)  # (cos θ, sin θ) of text baseline
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
                line_dir = tuple(line.get("dir", (1.0, 0.0)))  # (cos θ, sin θ)
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
                        direction  = line_dir,
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
    """
    Map original font to a built-in PDF font preserving bold and italic weight.
    PyMuPDF Base-14 aliases: helv/hebo/heit/hebi, cour/cobo/coob/cobi, tiro/tibo/tiit/tibi.
    """
    n = font_name.lower()
    is_bold   = bool(flags & 16) or any(k in n for k in ("bold", "demi", "black", "heavy"))
    is_italic = bool(flags & 2)  or any(k in n for k in ("italic", "oblique", "slant"))

    if any(k in n for k in ("mono", "courier", "cour", "consol", "code", "fixed")):
        if is_bold and is_italic: return "cobi"   # Courier-BoldOblique
        if is_bold:               return "cobo"   # Courier-Bold
        if is_italic:             return "coob"   # Courier-Oblique
        return "cour"

    if any(k in n for k in ("times", "serif", "georgia", "garamond")):
        if is_bold and is_italic: return "tibi"   # Times-BoldItalic
        if is_bold:               return "tibo"   # Times-Bold
        if is_italic:             return "tiit"   # Times-Italic
        return "tiro"

    # Helvetica (default)
    if is_bold and is_italic: return "hebi"       # Helvetica-BoldOblique
    if is_bold:               return "hebo"       # Helvetica-Bold
    if is_italic:             return "heit"       # Helvetica-Oblique
    return "helv"


def _fit_size(
    text: str,
    font: str,
    orig_size: float,
    rect: fitz.Rect,
    min_size: float = 4.0,
    font_file: str | None = None,
) -> float:
    """Shrink font size until the text fits within the span's width (single-font path)."""
    size = orig_size
    while size >= min_size:
        try:
            if font_file:
                tw = fitz.get_text_length(text, fontfile=font_file, fontname=font, fontsize=size)
            else:
                tw = fitz.get_text_length(text, fontname=font, fontsize=size)
        except Exception:
            tw = len(text) * size * 0.55
        if tw <= rect.width:
            break
        size -= 0.5
    return max(size, min_size)


def _fit_size_mixed(
    text: str,
    unicode_font: str | None,
    unicode_alias: str,
    latin_font: str,
    orig_size: float,
    rect: fitz.Rect,
    min_size: float = 7.0,
) -> float:
    """
    Shrink font size for mixed-script text.
    Sums per-run widths using each run's actual font so the estimate is accurate.
    A 5 % slack (1.05×) prevents crushing slightly-wider translations to tiny sizes.
    Never goes below min_size to keep text readable.
    """
    runs = _split_text_runs(text)
    size = orig_size
    limit = rect.width * 1.05   # 5 % slack — better a tiny overhang than 4 pt text
    while size >= min_size:
        total = 0.0
        for needs_unicode, run_text in runs:
            if needs_unicode and unicode_font:
                try:
                    total += fitz.get_text_length(
                        run_text, fontfile=unicode_font, fontname=unicode_alias, fontsize=size,
                    )
                except Exception:
                    total += len(run_text) * size * 0.6
            else:
                try:
                    total += fitz.get_text_length(run_text, fontname=latin_font, fontsize=size)
                except Exception:
                    total += len(run_text) * size * 0.55
        if total <= limit:
            break
        size -= 0.5
    return max(size, min_size)


def _render_page_pixmap(page: fitz.Page) -> fitz.Pixmap:
    """Render page at half resolution before modifications — used for background sampling."""
    return page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5), colorspace=fitz.csRGB, annots=False)


def _sample_bg_pixmap(
    pix: fitz.Pixmap,
    rect: fitz.Rect,
) -> tuple[float, float, float]:
    """
    Sample background colour from the pre-modification page pixmap.
    Collects a grid of pixels across the span's bounding box and returns
    the median RGB (by luminance) — robust to isolated dark text pixels
    that would bias a simple average or lightest-wins heuristic.
    """
    scale = 0.5
    x0 = max(0, int(rect.x0 * scale))
    x1 = min(pix.width  - 1, int(rect.x1 * scale))
    y0 = max(0, int(rect.y0 * scale))
    y1 = min(pix.height - 1, int(rect.y1 * scale))

    samples: list[tuple[float, float, float]] = []
    xs = [x0, (x0 + x1) // 2, x1]
    ys = [y0, (y0 + y1) // 2, y1]
    for xi in xs:
        for yi in ys:
            try:
                p = pix.pixel(xi, yi)
                samples.append((p[0] / 255.0, p[1] / 255.0, p[2] / 255.0))
            except Exception:
                pass

    if not samples:
        return REDACT_COLOR
    # Sort by luminance and take the median — text ink (dark) is sorted to the low end
    samples.sort(key=lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
    return samples[len(samples) // 2]


def _build_bg_map(page: fitz.Page) -> list[tuple[fitz.Rect, tuple[float, float, float]]]:
    """
    Snapshot the filled drawing paths on a page BEFORE we modify it.
    Returns a list of (rect, rgb) in PDF stream order (last entry = topmost drawn).
    We skip hairline or degenerate paths that can't be backgrounds.
    """
    bg: list[tuple[fitz.Rect, tuple[float, float, float]]] = []
    for d in page.get_drawings():
        fill = d.get("fill")
        rect = d.get("rect")
        if not fill or not rect or len(fill) < 3:
            continue
        r = fitz.Rect(rect)
        if r.width < 2 or r.height < 2:
            continue
        bg.append((r, (float(fill[0]), float(fill[1]), float(fill[2]))))
    return bg


def _find_bg_color(
    bg_map: list[tuple[fitz.Rect, tuple[float, float, float]]],
    pt: fitz.Point,
) -> tuple[float, float, float]:
    """
    Return the fill colour of the topmost drawing that contains pt.
    Iterates in reverse stream order so the last-drawn (visually topmost) wins.
    Falls back to white if no drawing covers pt.
    """
    for rect, color in reversed(bg_map):
        if rect.contains(pt):
            return color
    return REDACT_COLOR  # white


def rebuild_pdf(doc: fitz.Document, pages: list[PageSpans], target_lang: str = "") -> fitz.Document:
    """
    For each translated span:
      1. Erase original text using the detected background colour (not a hardcoded white)
      2. Insert translated text at the exact baseline with matching style

    Mixed-script text (e.g. Hindi body with Latin abbreviations like TML/JLR) is
    split into per-script runs so each segment is rendered with the correct font.
    """
    # Pre-resolve regular/bold unicode fonts for this language (cached after first call)
    unicode_font_r = _get_unicode_font(target_lang, is_bold=False) if target_lang else None
    unicode_font_b = _get_unicode_font(target_lang, is_bold=True)  if target_lang else None
    if target_lang and not unicode_font_r:
        log.info("No unicode font found for '%s', falling back to built-in Latin fonts.", target_lang)

    for page_data in pages:
        page = doc[page_data.page_num]
        # Snapshot background BEFORE any modifications (both vector and raster)
        page_pix = _render_page_pixmap(page)
        bg_map   = _build_bg_map(page)

        for span in page_data.spans:
            if not span.translated or span.translated == span.text:
                continue

            # Leave spans whose original text contains Private Use Area characters
            # (U+E000–U+F8FF) — these are custom brand-logo/icon glyphs embedded in
            # the original PDF font. No standard font can render them, so erasing the
            # span and replacing with our font produces □ boxes. Skipping preserves
            # the original custom-font rendering.
            if any(0xE000 <= ord(ch) <= 0xF8FF for ch in span.text):
                continue

            n = span.font_name.lower()
            is_bold = bool(span.font_flags & 16) or any(
                k in n for k in ("bold", "demi", "black", "heavy")
            )

            color      = _color_from_int(span.color)
            latin_font = _resolve_font(span.font_name, span.font_flags)

            # Determine whether the translated text is "mixed" (contains both
            # Devanagari/non-Latin AND Basic-Latin characters like TGS, USP, etc.)
            _non_combining = lambda ch: unicodedata.category(ch) not in ('Mn', 'Mc', 'Me', 'Cf')
            trans_chars = [ch for ch in span.translated if _non_combining(ch) and ch.strip()]
            _has_nonlatin_chars = any(_needs_unicode_font(ch) for ch in trans_chars)
            _has_latin_chars    = any(not _needs_unicode_font(ch) for ch in trans_chars)
            is_mixed = _has_nonlatin_chars and _has_latin_chars

            # Font alias selection:
            # • Pure Devanagari + bold → NotoSansDevanagari-Bold (proper bold, no Latin needed)
            # • Mixed Devanagari+Latin → Lohit (regular weight, but covers BOTH scripts —
            #   NotoSansDevanagari-Bold has NO Basic Latin, so TGS/USP would render as □)
            # • Regular weight → Lohit / NotoSansDevanagari-Regular
            if is_bold and unicode_font_b and not is_mixed:
                unicode_font  = unicode_font_b   # pure Devanagari bold — no Latin needed
                unicode_alias = "F1"
            else:
                unicode_font  = unicode_font_r   # Lohit: Devanagari + Latin in one font
                unicode_alias = "F0"

            rotate     = _dir_to_rotate(span.direction)
            is_vertical = rotate in (90, 270)

            # Detect background colour before erasing
            span_centre = fitz.Point(
                (span.rect.x0 + span.rect.x1) / 2,
                (span.rect.y0 + span.rect.y1) / 2,
            )
            bg_color = _find_bg_color(bg_map, span_centre)
            if bg_color == REDACT_COLOR:
                bg_color = _sample_bg_pixmap(page_pix, span.rect)

            erase_rect = fitz.Rect(
                span.rect.x0 - 1, span.rect.y0 - 1,
                span.rect.x1 + 1, span.rect.y1 + 1,
            )
            shape = page.new_shape()
            shape.draw_rect(erase_rect)
            shape.finish(color=None, fill=bg_color, width=0)
            shape.commit()

            # ── Insertion strategy ──────────────────────────────────────────────
            # Detect whether the translated text contains any non-Latin characters
            has_nonlatin = unicode_font and any(
                _needs_unicode_font(ch)
                for ch in span.translated
                if unicodedata.category(ch) not in ('Mn', 'Mc', 'Me', 'Cf')
            )

            if has_nonlatin:
                # Non-Latin script: use insert_textbox — it manages its own cursor
                # arithmetic internally, avoiding the accumulated measurement errors
                # that plague manual per-run placement of shaped Devanagari glyphs.
                tb_rect = span.rect
                orig_size = span.font_size
                min_size  = 7.0
                placed    = False
                for sz in (s / 2 for s in range(int(orig_size * 2), int(min_size * 2) - 1, -1)):
                    try:
                        rc = page.insert_textbox(
                            tb_rect, span.translated,
                            fontfile=unicode_font,
                            fontname=unicode_alias,
                            fontsize=sz,
                            color=color,
                            rotate=rotate,
                            align=fitz.TEXT_ALIGN_LEFT,
                        )
                        if rc >= 0:   # ≥0 → all chars placed
                            placed = True
                            break
                    except Exception as exc:
                        log.warning("textbox p%d sz=%.1f: %s", page_data.page_num, sz, exc)
                        break
                if not placed:
                    # Fall back: insert at minimum size regardless of fit
                    try:
                        page.insert_textbox(
                            tb_rect, span.translated,
                            fontfile=unicode_font,
                            fontname=unicode_alias,
                            fontsize=min_size,
                            color=color,
                            rotate=rotate,
                            align=fitz.TEXT_ALIGN_LEFT,
                        )
                    except Exception as exc:
                        log.warning("textbox fallback p%d: %s", page_data.page_num, exc)
            else:
                # Latin-only: precise baseline placement with insert_text
                pt   = fitz.Point(span.origin[0], span.origin[1])
                size = _fit_size(span.translated, latin_font, span.font_size, span.rect)
                try:
                    page.insert_text(
                        pt, span.translated, fontname=latin_font,
                        fontsize=size, color=color, rotate=rotate,
                    )
                except Exception:
                    try:
                        page.insert_text(
                            pt, span.translated, fontname="helv",
                            fontsize=size, color=color, rotate=rotate,
                        )
                    except Exception as exc:
                        log.warning("insert_text p%d: %s", page_data.page_num, exc)

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
