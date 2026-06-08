"""
app/core/engine.py
==================
Production-grade PDF translation engine.
Extracts text spans with full styling → translates in parallel → reconstructs PDF.
All images, positions, colors, and font metrics are preserved.

Text deletion: uses PDF redaction (removes text from content stream) rather than
painting rectangles over the original text, giving clean output with smaller files.
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
MAX_PAGES       = int(os.environ.get("MAX_PAGES", 500))
REDACT_COLOR    = (1.0, 1.0, 1.0)  # white fallback

# Codepoints that indicate a span was extracted from a custom/non-Unicode-encoded font.
# These characters never appear in real body text; their presence means PyMuPDF's
# extraction could not map glyphs to real Unicode (no ToUnicode CMap in the font).
# Inserting a translation over such spans produces garbled glyphs, so we skip them.
_GARBAGE_CODEPOINTS: frozenset[int] = (
    frozenset(range(0x00, 0x09))   # C0 control (NUL … BS; keep TAB 0x09, LF 0x0A, CR 0x0D)
    | frozenset(range(0x0E, 0x20)) # C0 control (SO … US)
    | frozenset(range(0x80, 0xA0)) # C1 control / Win-1252 legacy range — custom PDF fonts
                                   # frequently map decorative glyphs here; PyMuPDF yields
                                   # these codepoints when the font has no ToUnicode CMap.
    | frozenset(range(0xD800, 0xE000))  # Surrogates — invalid in text, corrupt extraction
    | {0x7F,                        # DEL
       0xFFFD,                      # Replacement character (U+FFFD)
       0xFFFE, 0xFFFF}              # Unicode non-characters
)

# PyMuPDF redaction constants (with getattr fallback for older builds)

# ── Unicode font paths ─────────────────────────────────────────────────────────

_NOTO_TTF   = Path("/usr/share/fonts/truetype/noto")
_NOTO_OTF   = Path("/usr/share/fonts/opentype/noto")
_LOHIT_BASE = Path("/usr/share/fonts/truetype")

# Maps target language → regular ("r") and bold ("b") font file candidates.
# First existing path wins.  Lohit fonts are listed first because they cover
# BOTH the Indic script AND Basic Latin in one TTF, so Latin abbreviations
# (EBITDA, JLR, TML …) render correctly without multi-run cursor arithmetic.
_LANG_FONT_CANDIDATES: dict[str, dict[str, list[str]]] = {

    # ── Devanagari script ──────────────────────────────────────────────────────
    # NotoSansDevanagari is listed first: it has superior HarfBuzz GSUB tables
    # for conjunct formation (e.g. ट्रे, क्ष, ज्ञ) and covers Basic Latin so
    # mixed Hindi+Latin text (EBITDA मार्जिन) renders correctly from one font.
    # OTF path is a fallback for systems that install Noto under opentype/.
    # Lohit-Devanagari is the final fallback (older systems / smaller footprint).
    "Hindi":    {"r": [str(_NOTO_TTF / "NotoSansDevanagari-Regular.ttf"),
                       str(_NOTO_OTF / "NotoSansDevanagari-Regular.ttf"),
                       str(_LOHIT_BASE / "lohit-devanagari/Lohit-Devanagari.ttf")],
                 "b": [str(_NOTO_TTF / "NotoSansDevanagari-Bold.ttf"),
                       str(_NOTO_OTF / "NotoSansDevanagari-Bold.ttf"),
                       str(_LOHIT_BASE / "lohit-devanagari/Lohit-Devanagari.ttf")]},
    "Marathi":  {"r": [str(_NOTO_TTF / "NotoSansDevanagari-Regular.ttf"),
                       str(_NOTO_OTF / "NotoSansDevanagari-Regular.ttf"),
                       str(_LOHIT_BASE / "lohit-devanagari/Lohit-Devanagari.ttf")],
                 "b": [str(_NOTO_TTF / "NotoSansDevanagari-Bold.ttf"),
                       str(_NOTO_OTF / "NotoSansDevanagari-Bold.ttf"),
                       str(_LOHIT_BASE / "lohit-devanagari/Lohit-Devanagari.ttf")]},
    "Nepali":   {"r": [str(_NOTO_TTF / "NotoSansDevanagari-Regular.ttf"),
                       str(_NOTO_OTF / "NotoSansDevanagari-Regular.ttf"),
                       str(_LOHIT_BASE / "lohit-devanagari/Lohit-Devanagari.ttf")],
                 "b": [str(_NOTO_TTF / "NotoSansDevanagari-Bold.ttf"),
                       str(_NOTO_OTF / "NotoSansDevanagari-Bold.ttf"),
                       str(_LOHIT_BASE / "lohit-devanagari/Lohit-Devanagari.ttf")]},
    "Sanskrit": {"r": [str(_NOTO_TTF / "NotoSansDevanagari-Regular.ttf"),
                       str(_NOTO_OTF / "NotoSansDevanagari-Regular.ttf"),
                       str(_LOHIT_BASE / "lohit-devanagari/Lohit-Devanagari.ttf")],
                 "b": [str(_NOTO_TTF / "NotoSansDevanagari-Bold.ttf"),
                       str(_NOTO_OTF / "NotoSansDevanagari-Bold.ttf"),
                       str(_LOHIT_BASE / "lohit-devanagari/Lohit-Devanagari.ttf")]},
    "Maithili": {"r": [str(_NOTO_TTF / "NotoSansDevanagari-Regular.ttf"),
                       str(_NOTO_OTF / "NotoSansDevanagari-Regular.ttf"),
                       str(_LOHIT_BASE / "lohit-devanagari/Lohit-Devanagari.ttf")],
                 "b": [str(_NOTO_TTF / "NotoSansDevanagari-Bold.ttf"),
                       str(_NOTO_OTF / "NotoSansDevanagari-Bold.ttf"),
                       str(_LOHIT_BASE / "lohit-devanagari/Lohit-Devanagari.ttf")]},
    "Bodo":     {"r": [str(_NOTO_TTF / "NotoSansDevanagari-Regular.ttf"),
                       str(_NOTO_OTF / "NotoSansDevanagari-Regular.ttf"),
                       str(_LOHIT_BASE / "lohit-devanagari/Lohit-Devanagari.ttf")],
                 "b": [str(_NOTO_TTF / "NotoSansDevanagari-Bold.ttf"),
                       str(_NOTO_OTF / "NotoSansDevanagari-Bold.ttf"),
                       str(_LOHIT_BASE / "lohit-devanagari/Lohit-Devanagari.ttf")]},
    "Dogri":    {"r": [str(_NOTO_TTF / "NotoSansDevanagari-Regular.ttf"),
                       str(_NOTO_OTF / "NotoSansDevanagari-Regular.ttf"),
                       str(_LOHIT_BASE / "lohit-devanagari/Lohit-Devanagari.ttf")],
                 "b": [str(_NOTO_TTF / "NotoSansDevanagari-Bold.ttf"),
                       str(_NOTO_OTF / "NotoSansDevanagari-Bold.ttf"),
                       str(_LOHIT_BASE / "lohit-devanagari/Lohit-Devanagari.ttf")]},
    "Konkani":  {"r": [str(_NOTO_TTF / "NotoSansDevanagari-Regular.ttf"),
                       str(_NOTO_OTF / "NotoSansDevanagari-Regular.ttf"),
                       str(_LOHIT_BASE / "lohit-devanagari/Lohit-Devanagari.ttf")],
                 "b": [str(_NOTO_TTF / "NotoSansDevanagari-Bold.ttf"),
                       str(_NOTO_OTF / "NotoSansDevanagari-Bold.ttf"),
                       str(_LOHIT_BASE / "lohit-devanagari/Lohit-Devanagari.ttf")]},

    # ── Bengali / Assamese script ──────────────────────────────────────────────
    # Noto primary (superior HarfBuzz shaping), Lohit fallback.
    "Bengali":  {"r": [str(_NOTO_TTF / "NotoSansBengali-Regular.ttf"),
                       str(_LOHIT_BASE / "lohit-bengali/Lohit-Bengali.ttf")],
                 "b": [str(_NOTO_TTF / "NotoSansBengali-Bold.ttf"),
                       str(_LOHIT_BASE / "lohit-bengali/Lohit-Bengali.ttf")]},
    "Assamese": {"r": [str(_NOTO_TTF / "NotoSansBengali-Regular.ttf"),
                       str(_LOHIT_BASE / "lohit-assamese/Lohit-Assamese.ttf"),
                       str(_LOHIT_BASE / "lohit-bengali/Lohit-Bengali.ttf")],
                 "b": [str(_NOTO_TTF / "NotoSansBengali-Bold.ttf"),
                       str(_LOHIT_BASE / "lohit-bengali/Lohit-Bengali.ttf")]},

    # ── Tamil script ───────────────────────────────────────────────────────────
    "Tamil":    {"r": [str(_NOTO_TTF / "NotoSansTamil-Regular.ttf"),
                       str(_LOHIT_BASE / "lohit-tamil/Lohit-Tamil.ttf"),
                       str(_LOHIT_BASE / "lohit-tamil-classical/Lohit-Tamil-Classical.ttf")],
                 "b": [str(_NOTO_TTF / "NotoSansTamil-Bold.ttf"),
                       str(_LOHIT_BASE / "lohit-tamil/Lohit-Tamil.ttf")]},

    # ── Telugu script ──────────────────────────────────────────────────────────
    "Telugu":   {"r": [str(_NOTO_TTF / "NotoSansTelugu-Regular.ttf"),
                       str(_LOHIT_BASE / "lohit-telugu/Lohit-Telugu.ttf")],
                 "b": [str(_NOTO_TTF / "NotoSansTelugu-Bold.ttf"),
                       str(_LOHIT_BASE / "lohit-telugu/Lohit-Telugu.ttf")]},

    # ── Kannada script ─────────────────────────────────────────────────────────
    "Kannada":  {"r": [str(_NOTO_TTF / "NotoSansKannada-Regular.ttf"),
                       str(_LOHIT_BASE / "lohit-kannada/Lohit-Kannada.ttf")],
                 "b": [str(_NOTO_TTF / "NotoSansKannada-Bold.ttf"),
                       str(_LOHIT_BASE / "lohit-kannada/Lohit-Kannada.ttf")]},

    # ── Malayalam script ───────────────────────────────────────────────────────
    "Malayalam":{"r": [str(_NOTO_TTF / "NotoSansMalayalam-Regular.ttf"),
                       str(_LOHIT_BASE / "lohit-malayalam/Lohit-Malayalam.ttf")],
                 "b": [str(_NOTO_TTF / "NotoSansMalayalam-Bold.ttf"),
                       str(_LOHIT_BASE / "lohit-malayalam/Lohit-Malayalam.ttf")]},

    # ── Gujarati script ────────────────────────────────────────────────────────
    "Gujarati": {"r": [str(_NOTO_TTF / "NotoSansGujarati-Regular.ttf"),
                       str(_LOHIT_BASE / "lohit-gujarati/Lohit-Gujarati.ttf")],
                 "b": [str(_NOTO_TTF / "NotoSansGujarati-Bold.ttf"),
                       str(_LOHIT_BASE / "lohit-gujarati/Lohit-Gujarati.ttf")]},

    # ── Gurmukhi script (Punjabi) ──────────────────────────────────────────────
    "Punjabi":  {"r": [str(_NOTO_TTF / "NotoSansGurmukhi-Regular.ttf"),
                       str(_LOHIT_BASE / "lohit-punjabi/Lohit-Gurmukhi.ttf")],
                 "b": [str(_NOTO_TTF / "NotoSansGurmukhi-Bold.ttf"),
                       str(_LOHIT_BASE / "lohit-punjabi/Lohit-Gurmukhi.ttf")]},

    # ── Odia (Oriya) script ────────────────────────────────────────────────────
    "Odia":     {"r": [str(_NOTO_TTF / "NotoSansOriya-Regular.ttf"),
                       str(_NOTO_TTF / "NotoSansOdiya-Regular.ttf"),
                       str(_LOHIT_BASE / "lohit-oriya/Lohit-Odia.ttf")],
                 "b": [str(_NOTO_TTF / "NotoSansOriya-Bold.ttf"),
                       str(_NOTO_TTF / "NotoSansOdiya-Bold.ttf"),
                       str(_LOHIT_BASE / "lohit-oriya/Lohit-Odia.ttf")]},

    # ── Arabic / Perso-Arabic script ──────────────────────────────────────────
    "Arabic":   {"r": [str(_NOTO_TTF / "NotoNaskhArabic-Regular.ttf")],
                 "b": [str(_NOTO_TTF / "NotoNaskhArabic-Bold.ttf")]},
    "Urdu":     {"r": [str(_NOTO_TTF / "NotoNaskhArabic-Regular.ttf")],
                 "b": [str(_NOTO_TTF / "NotoNaskhArabic-Bold.ttf")]},
    "Persian":  {"r": [str(_NOTO_TTF / "NotoNaskhArabic-Regular.ttf")],
                 "b": [str(_NOTO_TTF / "NotoNaskhArabic-Bold.ttf")]},
    "Sindhi":   {"r": [str(_NOTO_TTF / "NotoNaskhArabic-Regular.ttf")],
                 "b": [str(_NOTO_TTF / "NotoNaskhArabic-Bold.ttf")]},
    # Kashmiri is written in both Perso-Arabic (official) and Devanagari
    "Kashmiri": {"r": [str(_NOTO_TTF / "NotoNaskhArabic-Regular.ttf"),
                       str(_LOHIT_BASE / "lohit-devanagari/Lohit-Devanagari.ttf")],
                 "b": [str(_NOTO_TTF / "NotoNaskhArabic-Bold.ttf")]},

    # ── Other scripts ──────────────────────────────────────────────────────────
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

# font_path → (fitz.Archive, CSS family name)
# Re-using Archives across pages is safe — Archive objects are read-only after creation.
_font_archive_cache: dict[str, tuple] = {}


def _get_font_archive(font_path: str | None) -> tuple:
    """Return a cached (fitz.Archive, font-family-name), or (None, None) on failure.

    Uses the Archive constructor with the font file's parent directory — this avoids
    the Archive.add() signature differences across PyMuPDF 1.24.x builds (some builds
    don't accept 'name' as a keyword argument). Failures are cached so we only warn once.
    """
    if not font_path or not Path(font_path).exists():
        return None, None
    if font_path not in _font_archive_cache:
        try:
            arc = fitz.Archive(str(Path(font_path).parent))
            _font_archive_cache[font_path] = (arc, Path(font_path).stem)
        except Exception as exc:
            log.warning("Archive creation failed for %s: %s", font_path, exc)
            _font_archive_cache[font_path] = (None, None)  # cache failure to suppress repeats
    return _font_archive_cache[font_path]


def _get_unicode_font(target_lang: str, is_bold: bool = False) -> str | None:
    """Return path to a font for the target language and weight, or None.

    Accepts both ISO language codes ("hi") and language names ("Hindi").
    """
    lang_name = LANG_CODE_MAP.get(target_lang, target_lang)  # "hi" → "Hindi"
    key = (lang_name, is_bold)
    if key not in _font_cache:
        weight_key = "b" if is_bold else "r"
        candidates = _LANG_FONT_CANDIDATES.get(lang_name, {}).get(weight_key, [])
        found = next((p for p in candidates if Path(p).exists()), None)
        if is_bold and not found:
            # Fall back to regular weight for bold if no dedicated bold font
            regular = _LANG_FONT_CANDIDATES.get(lang_name, {}).get("r", [])
            found = next((p for p in regular if Path(p).exists()), None)
        _font_cache[key] = found
    return _font_cache[key]


# ── Mixed-script text helpers ──────────────────────────────────────────────────

# Unicode ranges whose glyphs are absent from Latin-only PDF built-in fonts.
_NON_LATIN_RANGES = (
    (0x0900, 0x097F),  # Devanagari (Hindi, Marathi, Sanskrit, Bodo, Dogri …)
    (0x0980, 0x09FF),  # Bengali / Assamese
    (0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Odia (Oriya)
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
    (0x0D80, 0x0DFF),  # Sinhala
    (0x1C50, 0x1C7F),  # Ol Chiki (Santali)
    (0x0600, 0x06FF),  # Arabic / Urdu / Sindhi / Kashmiri
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
    For each page, true-delete the original text via PDF redaction (removes operators
    from the content stream), then insert the translated text at the exact baseline.

    Processing is batched per page:
      Phase 1 — collect spans eligible for translation
      Phase 2 — draw_rect() to cover original text (background colour matched from
                page drawings; white for body text, matched colour for coloured headers)
      Phase 3 — insert translated text at original baseline positions

    Bold: pure Indic bold → NotoSans*-Bold (real bold weight).
          Mixed Indic+Latin: Lohit regular (covers both scripts in one TTF).
    """
    unicode_font_r = _get_unicode_font(target_lang, is_bold=False) if target_lang else None
    unicode_font_b = _get_unicode_font(target_lang, is_bold=True)  if target_lang else None
    if target_lang and not unicode_font_r:
        log.info("No unicode font found for '%s', falling back to built-in Latin fonts.", target_lang)

    _is_combining = lambda ch: unicodedata.category(ch) in ('Mn', 'Mc', 'Me', 'Cf')

    for page_data in pages:
        page = doc[page_data.page_num]

        # ── Phase 1: collect spans eligible for translation ────────────────────
        pending: list[TextSpan] = []

        for span in page_data.spans:
            if not span.translated or span.translated == span.text:
                continue
            # Private Use Area glyphs (U+E000–U+F8FF) are custom brand/icon glyphs
            # embedded in the original font — no standard font can reproduce them.
            if any(0xE000 <= ord(ch) <= 0xF8FF for ch in span.text):
                continue
            # Control/special characters in the ORIGINAL text mean the PDF's font
            # has no proper Unicode map.  The extracted text is garbage and the
            # "translation" of garbage is also garbage, producing garbled glyphs.
            # Skip and leave the original rendering intact.
            if any(ord(ch) in _GARBAGE_CODEPOINTS for ch in span.text):
                continue
            # Also guard against garbage that slipped through to the translation.
            # PUA or control chars in the translated string means the LLM echoed
            # custom glyph codepoints back — inserting them produces the same blobs.
            t = span.translated
            if (any(ord(ch) in _GARBAGE_CODEPOINTS for ch in t)
                    or any(0xE000 <= ord(ch) <= 0xF8FF for ch in t)):
                continue
            pending.append(span)

        if not pending:
            continue

        # Deduplicate: two spans with identical text whose bounding boxes overlap or
        # whose y-midpoints are within 5pt of each other on the same line represent
        # the same visual element (e.g. a bold "BUY" badge AND "BUY" inside the
        # adjacent sentence span). Inserting both translations produces "खरीदें खरीदें".
        # Keep the first occurrence; subsequent near-duplicates are dropped.
        deduped: list[TextSpan] = []
        for span in pending:
            mid_y = (span.rect.y0 + span.rect.y1) * 0.5
            is_dup = any(
                s.text == span.text
                and abs((s.rect.y0 + s.rect.y1) * 0.5 - mid_y) < 5.0
                and not (s.rect & fitz.Rect(
                    span.rect.x0 - 3, span.rect.y0 - 3,
                    span.rect.x1 + 3, span.rect.y1 + 3,
                )).is_empty
                for s in deduped
            )
            if not is_dup:
                deduped.append(span)
        pending = deduped

        # ── Phase 2: cover original text with background-matched rectangles ─────
        # PDF redaction (add_redact_annot + apply_redactions) only processes the
        # page's own content stream.  Text rendered through Form XObjects — common
        # in InDesign/Illustrator-generated PDFs — lives in a separate sub-stream
        # that the redaction engine never enters, so the original glyphs stay visible
        # beneath our inserted translation and produce dark "blob" artefacts.
        #
        # draw_rect() appends a new graphics command to the END of the page stream,
        # so it paints on top of ALL existing content (including XObject output),
        # guaranteeing visual erasure of the original text regardless of how it was
        # encoded.
        #
        # Background colour matching: we query the page's own vector drawings via
        # get_drawings() to find any coloured filled shape that underlies the span.
        # If one is found we match it (so a white rect on a blue bar looks correct);
        # otherwise we use white (the dominant background in body text areas).
        # Collect coloured vector backgrounds for colour matching.
        _colored_bgs: list[tuple[fitz.Rect, tuple]] = []
        for d in page.get_drawings():
            fill = d.get("fill")
            if fill and not all(c > 0.92 for c in fill[:3]):
                try:
                    _colored_bgs.append((fitz.Rect(d["rect"]), tuple(fill[:3])))
                except Exception:
                    pass

        _PIX_SCALE = 0.5  # 50 % scale is fast and accurate enough for colour sampling

        # Pre-render the page pixmap BEFORE any draw_rect modifications so that
        # background colour sampling always reflects the original page content —
        # including text rendered through Form XObjects (coloured side strips,
        # ICICI-style vertical banners) that are invisible to get_drawings().
        _pix = page.get_pixmap(matrix=fitz.Matrix(_PIX_SCALE, _PIX_SCALE))

        def _bg_at(rect: fitz.Rect) -> tuple:
            area = rect.get_area()
            if area <= 0:
                return (1.0, 1.0, 1.0)
            best_ov, best_col = 0.0, (1.0, 1.0, 1.0)
            for bg_r, col in _colored_bgs:
                inter = bg_r & rect
                if inter.is_empty:
                    continue
                ov = inter.get_area() / area
                if ov > best_ov:
                    best_ov, best_col = ov, col
            return best_col if best_ov > 0.2 else (1.0, 1.0, 1.0)

        def _sample_bg(rect: fitz.Rect) -> tuple[tuple, float]:
            """
            Sample background colour from the pre-rendered pixmap.

            Probes rows ABOVE and BELOW the text span (not within it) so that
            text-ink pixels don't contaminate the result.  Returns
            (median_rgb, luminance_variance).  High variance means a gradient
            or complex background; low variance means a solid colour.
            """
            scale = _PIX_SCALE
            px0 = max(0, int(rect.x0 * scale))
            px1 = min(_pix.width  - 1, int(rect.x1 * scale))
            py0 = int(rect.y0 * scale)
            py1 = int(rect.y1 * scale)
            h   = max(1, py1 - py0)

            # x positions: left edge, centre, right edge
            xs = [px0, (px0 + px1) // 2, px1] if px1 > px0 else [px0]

            # Primary probes: half-a-line-height above and below (ink-free rows)
            probe_ys: list[int] = []
            above = py0 - max(2, h // 2)
            below = py1 + max(2, h // 2)
            if 0 <= above < _pix.height:
                probe_ys.append(above)
            if 0 <= below < _pix.height:
                probe_ys.append(below)

            # Fallback probe rows: within the span (may include ink)
            fallback_ys = [py0, (py0 + py1) // 2, py1]

            samples: list[tuple[float, float, float]] = []
            for yi in (probe_ys or fallback_ys):
                for xi in xs:
                    if 0 <= xi < _pix.width and 0 <= yi < _pix.height:
                        try:
                            p = _pix.pixel(xi, yi)
                            samples.append((p[0] / 255.0, p[1] / 255.0, p[2] / 255.0))
                        except Exception:
                            pass

            # If above/below probes gave nothing, fall back to in-span sampling
            if not samples:
                for yi in fallback_ys:
                    for xi in xs:
                        if 0 <= xi < _pix.width and 0 <= yi < _pix.height:
                            try:
                                p = _pix.pixel(xi, yi)
                                samples.append((p[0] / 255.0, p[1] / 255.0, p[2] / 255.0))
                            except Exception:
                                pass

            if not samples:
                return (1.0, 1.0, 1.0), 0.0

            lums  = [0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] for c in samples]
            mean  = sum(lums) / len(lums)
            var   = sum((l - mean) ** 2 for l in lums) / len(lums)

            samples.sort(key=lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
            return samples[len(samples) // 2], var

        # active = spans we will erase + replace with translated text
        active: list[TextSpan] = []
        _has_image_redacts = False
        for span in pending:
            bg         = _bg_at(span.rect)
            erase_rect = fitz.Rect(
                span.rect.x0 - 2, span.rect.y0 - 2,
                span.rect.x1 + 2, span.rect.y1 + 2,
            )
            if bg != (1.0, 1.0, 1.0):
                # Solid coloured vector rect (detected by get_drawings) → paint it.
                page.draw_rect(erase_rect, fill=bg, color=None)
            else:
                # No vector rect → inspect the pre-rendered pixmap.
                sampled, variance = _sample_bg(span.rect)
                lum = 0.299 * sampled[0] + 0.587 * sampled[1] + 0.114 * sampled[2]
                if lum > 0.90:
                    # Effectively white background (body text area).
                    page.draw_rect(erase_rect, fill=(1.0, 1.0, 1.0), color=None)
                elif variance < 0.008:
                    # Solid non-white background (Form XObject coloured strip,
                    # e.g. ICICI amber sidebar) — draw solid rect with sampled colour.
                    page.draw_rect(erase_rect, fill=sampled, color=None)
                else:
                    # Non-uniform / gradient background (hero section, photo banner)
                    # — transparent redaction removes original text without painting
                    # a colour patch that would clash with the gradient.
                    page.add_redact_annot(erase_rect, fill=None)
                    _has_image_redacts = True
            active.append(span)

        # Apply transparent redactions for image-background spans before inserting text
        if _has_image_redacts:
            page.apply_redactions(images=0, graphics=0)

        # ── Phase 3: insert translated texts at original baseline positions ─────
        for span in active:
            # NFC normalization is required for Devanagari (and other Indic scripts)
            # to form conjuncts correctly.  Without it, characters in NFD form are
            # inserted as separate codepoints and HarfBuzz shaping doesn't fire,
            # producing individual unjoined glyphs that look garbled.
            translated = unicodedata.normalize("NFC", span.translated).strip()
            if not translated:
                continue

            n       = span.font_name.lower()
            is_bold = bool(span.font_flags & 16) or any(
                k in n for k in ("bold", "demi", "black", "heavy")
            )
            color      = _color_from_int(span.color)
            latin_font = _resolve_font(span.font_name, span.font_flags)

            trans_chars  = [ch for ch in translated if not _is_combining(ch) and ch.strip()]
            has_nonlatin = any(_needs_unicode_font(ch) for ch in trans_chars)
            has_latin    = any(not _needs_unicode_font(ch) for ch in trans_chars)
            is_mixed     = has_nonlatin and has_latin

            # Pure Indic bold → use real bold font (NotoSans*-Bold).
            # Mixed Indic+Latin: use Lohit regular (covers both scripts).
            if is_bold and unicode_font_b and not is_mixed:
                unicode_font  = unicode_font_b
                unicode_alias = "F1"
            else:
                unicode_font  = unicode_font_r
                unicode_alias = "F0"

            rotate = _dir_to_rotate(span.direction)
            # For rotated text the constraint dimension flips: width ↔ height
            avail = span.rect.height if rotate in (90, 270) else span.rect.width

            if has_nonlatin and unicode_font:
                min_size = 7.0
                size     = span.font_size
                while size > min_size:
                    try:
                        tw = fitz.get_text_length(
                            translated, fontfile=unicode_font,
                            fontname=unicode_alias, fontsize=size,
                        )
                    except Exception:
                        tw = len(translated) * size * 0.55
                    if tw <= avail * 1.1:
                        break
                    size -= 0.5
                size = max(size, min_size)

                # If translated text still overflows at minimum size, skip.
                # Inserting text wider than 2× the available span width causes it to
                # bleed into adjacent spans, stacking into unreadable dark blobs.
                try:
                    tw_final = fitz.get_text_length(
                        translated, fontfile=unicode_font,
                        fontname=unicode_alias, fontsize=size,
                    )
                except Exception:
                    tw_final = len(translated) * size * 0.55
                if tw_final > avail * 2.0:
                    log.debug(
                        "skip overflow span p%d (tw=%.1f avail=%.1f)",
                        page_data.page_num, tw_final, avail,
                    )
                    continue

                arc, ff = _get_font_archive(unicode_font) if rotate == 0 else (None, None)
                if arc is not None:
                    # insert_htmlbox routes through MuPDF's Story + HarfBuzz engine,
                    # which applies GSUB/GPOS lookups so Devanagari conjuncts
                    # (e.g. ट् + र → ट्र) form correctly in the stored glyph stream.
                    # insert_text bypasses shaping and stores raw codepoint→GID
                    # mappings, producing unjoined individual glyphs.
                    r_i = int(color[0] * 255)
                    g_i = int(color[1] * 255)
                    b_i = int(color[2] * 255)
                    css = (
                        f"@font-face{{font-family:'{ff}';"
                        f"src:url('{Path(unicode_font).name}');}}"
                        f"p{{font-family:'{ff}';font-size:{size}pt;"
                        f"color:rgb({r_i},{g_i},{b_i});"
                        f"margin:0;padding:0;white-space:nowrap;}}"
                    )
                    try:
                        page.insert_htmlbox(
                            span.rect, f"<p>{translated}</p>",
                            css=css, archive=arc,
                        )
                    except Exception as exc:
                        log.warning("insert_htmlbox p%d: %s", page_data.page_num, exc)
                        try:
                            page.insert_text(
                                fitz.Point(span.origin[0], span.origin[1]),
                                translated, fontfile=unicode_font,
                                fontname=unicode_alias, fontsize=size,
                                color=color, rotate=0,
                                render_mode=0, border_width=0.0,
                            )
                        except Exception as e2:
                            log.warning("unicode insert_text fallback p%d: %s",
                                        page_data.page_num, e2)
                else:
                    # Rotated text, or htmlbox archive unavailable:
                    # insert_text is the fallback (no HarfBuzz shaping, but functional).
                    try:
                        page.insert_text(
                            fitz.Point(span.origin[0], span.origin[1]),
                            translated, fontfile=unicode_font,
                            fontname=unicode_alias, fontsize=size,
                            color=color, rotate=rotate,
                            render_mode=0, border_width=0.0,
                        )
                    except Exception as exc:
                        log.warning("unicode insert_text p%d: %s", page_data.page_num, exc)

            else:
                pt   = fitz.Point(span.origin[0], span.origin[1])
                # Pass a rect with the correct available dimension for vertical text
                fit_rect = (
                    fitz.Rect(span.rect.x0, span.rect.y0,
                               span.rect.x0 + span.rect.height, span.rect.y1)
                    if rotate in (90, 270) else span.rect
                )
                size = _fit_size(translated, latin_font, span.font_size, fit_rect)
                try:
                    tw_final = fitz.get_text_length(translated, fontname=latin_font, fontsize=size)
                except Exception:
                    tw_final = len(translated) * size * 0.55
                if tw_final > avail * 2.0:
                    log.debug(
                        "skip overflow span (latin) p%d (tw=%.1f avail=%.1f)",
                        page_data.page_num, tw_final, avail,
                    )
                    continue
                try:
                    page.insert_text(
                        pt, translated, fontname=latin_font,
                        fontsize=size, color=color, rotate=rotate,
                    )
                except Exception:
                    try:
                        page.insert_text(
                            pt, translated, fontname="helv",
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

    if n_pages > MAX_PAGES:
        doc.close()
        raise ValueError(
            f"Document has {n_pages} pages; maximum allowed is {MAX_PAGES}. "
            "Split the document and resubmit."
        )

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
