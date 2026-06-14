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
class TextBlock:
    page_num:   int
    block_idx:  int
    text:       str        # combined text of all spans in the block (paragraph)
    rect:       fitz.Rect  # bounding box of the entire block
    spans:      list       # raw span dicts from get_text("dict") for font/color/origin
    translated: str = ""


@dataclass
class PageSpans:
    page_num: int
    spans:    list = field(default_factory=list)  # list[TextBlock]


def _dominant_span(spans: list) -> dict:
    """Return the span with the most non-whitespace characters (font/color representative)."""
    if not spans:
        return {}
    return max(spans, key=lambda s: len(s.get("text", "").strip()))


# ── Extraction ─────────────────────────────────────────────────────────────────

def extract_pages(doc: fitz.Document) -> list[PageSpans]:
    """
    Extract text style-groups from every page using PyMuPDF dict.

    Grouping strategy — LINE-level, not span-level:
      • Each PDF line is treated as an atomic unit.  Its dominant span (most
        non-whitespace chars) determines the line's style key (rounded font
        size + bold flag).  This keeps mixed-style inline content (e.g. bold
        term followed by a regular "/" separator) together on one line so the
        translator sees it as a single string.
      • Consecutive lines within the same block that share the same style key
        are merged into one TextBlock so the LLM sees full paragraphs.
      • A style change between consecutive lines (e.g. a bold heading line
        followed by regular body lines) creates a new TextBlock, allowing
        each to render with the correct weight and font size.
    """
    result: list[PageSpans] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_data = PageSpans(page_num=page_num)
        raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for b_idx, block in enumerate(raw.get("blocks", [])):
            if block.get("type") != 0:
                continue

            block_x0 = float(block["bbox"][0])
            block_x1 = float(block["bbox"][2])

            # ── Build one entry per line ───────────────────────────────────────
            line_entries: list[tuple] = []  # (style_key, line_text, spans, y0, y1)

            for line in block.get("lines", []):
                valid_spans: list[dict] = []
                parts:       list[str]  = []

                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    if (any(ord(ch) in _GARBAGE_CODEPOINTS for ch in text)
                            or any(0xE000 <= ord(ch) <= 0xF8FF for ch in text)):
                        continue
                    valid_spans.append(span)
                    parts.append(text)

                if not valid_spans:
                    continue

                line_text = " ".join(parts)

                # Style key from the dominant (longest) span in this line
                dom    = max(valid_spans, key=lambda s: len(s.get("text", "").strip()))
                ff     = int(dom.get("flags", 0))
                fn     = dom.get("font", "").lower()
                bold   = bool(ff & 16) or any(k in fn for k in ("bold", "demi", "black", "heavy"))
                sz_key = round(float(dom.get("size", 11.0)))

                # Y-extent from the line bbox (preferred) or span bboxes
                lb = line.get("bbox")
                if lb:
                    ly0, ly1 = float(lb[1]), float(lb[3])
                else:
                    yr = fitz.Rect(valid_spans[0]["bbox"])
                    for s in valid_spans[1:]:
                        if s.get("bbox"):
                            yr |= fitz.Rect(s["bbox"])
                    ly0, ly1 = yr.y0, yr.y1

                line_entries.append(((sz_key, bold), line_text, valid_spans, ly0, ly1))

            if not line_entries:
                continue

            # ── Merge consecutive lines that share the same style key ──────────
            groups: list[tuple] = []   # (key, text_lines, spans, y0, y1)
            cur_key    = line_entries[0][0]
            cur_lines  = [line_entries[0][1]]
            cur_spans  = list(line_entries[0][2])
            cur_y0     = line_entries[0][3]
            cur_y1     = line_entries[0][4]

            for key, lt, ls, ly0, ly1 in line_entries[1:]:
                if key == cur_key:
                    cur_lines.append(lt)
                    cur_spans.extend(ls)
                    cur_y1 = max(cur_y1, ly1)
                else:
                    groups.append((cur_key, cur_lines, cur_spans, cur_y0, cur_y1))
                    cur_key, cur_lines, cur_spans = key, [lt], list(ls)
                    cur_y0, cur_y1 = ly0, ly1
            groups.append((cur_key, cur_lines, cur_spans, cur_y0, cur_y1))

            # ── One TextBlock per style group ─────────────────────────────────
            for sub_idx, (_key, grp_lines, grp_spans, gy0, gy1) in enumerate(groups):
                grp_text = "\n".join(grp_lines).strip()
                if len(grp_text) < MIN_TEXT_LENGTH:
                    continue

                # Full block x-extent so translations have room; group's own y-extent
                grp_rect = fitz.Rect(block_x0, gy0, block_x1, gy1)

                page_data.spans.append(TextBlock(
                    page_num  = page_num,
                    block_idx = b_idx * 1000 + sub_idx,
                    text      = grp_text,
                    rect      = grp_rect,
                    spans     = grp_spans,
                ))

        result.append(page_data)

    total = sum(len(p.spans) for p in result)
    log.info(f"Extracted {total} text style-groups from {len(result)} pages.")
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
    For each page, erase original text blocks then insert translated text.

    Processing per page:
      Phase 1 — erase block bounding boxes (background-colour matched via vector
                drawings and pixmap sampling, same as before)
      Phase 2 — insert translated text using insert_htmlbox for all scripts:
                • Indic / non-Latin → unicode font archive + HarfBuzz shaping
                • Latin             → standard CSS font, no archive needed
                Both paths auto-wrap within the block rect and auto-scale font
                if text would overflow (scale_low=0.5 floor).
    """
    unicode_font_r = _get_unicode_font(target_lang, is_bold=False) if target_lang else None
    unicode_font_b = _get_unicode_font(target_lang, is_bold=True)  if target_lang else None
    if target_lang and not unicode_font_r:
        log.info("No unicode font found for '%s', falling back to built-in Latin fonts.", target_lang)

    _is_combining = lambda ch: unicodedata.category(ch) in ('Mn', 'Mc', 'Me', 'Cf')

    for page_data in pages:
        page = doc[page_data.page_num]

        # ── Collect eligible blocks ────────────────────────────────────────────
        pending: list = []
        for block in page_data.spans:
            if not block.translated or block.translated == block.text:
                continue
            t = block.translated
            if (any(ord(ch) in _GARBAGE_CODEPOINTS for ch in t)
                    or any(0xE000 <= ord(ch) <= 0xF8FF for ch in t)):
                continue
            pending.append(block)

        if not pending:
            continue

        # ── Snapshot coloured vector backgrounds ──────────────────────────────
        _colored_bgs: list[tuple[fitz.Rect, tuple]] = []
        for d in page.get_drawings():
            fill = d.get("fill")
            if fill and not all(c > 0.92 for c in fill[:3]):
                try:
                    _colored_bgs.append((fitz.Rect(d["rect"]), tuple(fill[:3])))
                except Exception:
                    pass

        _PIX_SCALE = 0.5
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

        def _sample_bg(rect: fitz.Rect) -> tuple:
            scale = _PIX_SCALE
            px0 = max(0, int(rect.x0 * scale))
            px1 = min(_pix.width  - 1, int(rect.x1 * scale))
            py0 = int(rect.y0 * scale)
            py1 = int(rect.y1 * scale)
            h   = max(1, py1 - py0)
            xs  = [px0, (px0 + px1) // 2, px1] if px1 > px0 else [px0]
            probe_ys: list[int] = []
            above = py0 - max(2, h // 2)
            below = py1 + max(2, h // 2)
            if 0 <= above < _pix.height:
                probe_ys.append(above)
            if 0 <= below < _pix.height:
                probe_ys.append(below)
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
            lums = [0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] for c in samples]
            mean = sum(lums) / len(lums)
            var  = sum((l - mean) ** 2 for l in lums) / len(lums)
            samples.sort(key=lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
            return samples[len(samples) // 2], var

        # ── Phase 1: erase block bounding boxes ───────────────────────────────
        _has_image_redacts = False
        active: list = []
        for block in pending:
            bg = _bg_at(block.rect)
            erase_rect = fitz.Rect(
                block.rect.x0 - 1, block.rect.y0 - 1,
                block.rect.x1 + 1, block.rect.y1 + 1,
            )
            if bg != (1.0, 1.0, 1.0):
                page.draw_rect(erase_rect, fill=bg, color=None)
            else:
                sampled, variance = _sample_bg(block.rect)
                lum = 0.299 * sampled[0] + 0.587 * sampled[1] + 0.114 * sampled[2]
                if lum > 0.90:
                    page.draw_rect(erase_rect, fill=(1.0, 1.0, 1.0), color=None)
                elif variance < 0.008:
                    page.draw_rect(erase_rect, fill=sampled, color=None)
                else:
                    page.add_redact_annot(erase_rect, fill=None)
                    _has_image_redacts = True
            active.append(block)

        if _has_image_redacts:
            page.apply_redactions(images=0, graphics=0)

        # ── Phase 2: insert translated text via htmlbox ────────────────────────
        for block in active:
            translated = unicodedata.normalize("NFC", block.translated).strip()
            if not translated:
                continue

            dom        = _dominant_span(block.spans)
            font_name  = dom.get("font", "helv")
            font_size  = float(dom.get("size", 11.0))
            font_flags = int(dom.get("flags", 0))
            color      = _color_from_int(int(dom.get("color", 0)))
            r_i        = int(color[0] * 255)
            g_i        = int(color[1] * 255)
            b_i        = int(color[2] * 255)

            n        = font_name.lower()
            is_bold  = bool(font_flags & 16) or any(k in n for k in ("bold", "demi", "black", "heavy"))
            is_italic = bool(font_flags & 2)  or any(k in n for k in ("italic", "oblique", "slant"))

            trans_chars  = [ch for ch in translated if not _is_combining(ch) and ch.strip()]
            has_nonlatin = any(_needs_unicode_font(ch) for ch in trans_chars)

            if has_nonlatin and (unicode_font_r or unicode_font_b):
                if is_bold and unicode_font_b:
                    unicode_font  = unicode_font_b
                    unicode_alias = "F1"
                    weight        = "bold"
                else:
                    unicode_font  = unicode_font_r
                    unicode_alias = "F0"
                    weight        = "normal"
                arc, ff = _get_font_archive(unicode_font)
                if arc is not None:
                    css = (
                        f"@font-face{{font-family:'{ff}';"
                        f"src:url('{Path(unicode_font).name}');"
                        f"font-weight:{weight};}}"
                        f"p{{font-family:'{ff}';font-size:{font_size}pt;"
                        f"font-weight:{weight};"
                        f"line-height:1.4;"
                        f"color:rgb({r_i},{g_i},{b_i});"
                        f"margin:0;padding:0;white-space:normal;}}"
                    )
                    try:
                        page.insert_htmlbox(
                            block.rect, f"<p>{translated}</p>",
                            css=css, archive=arc, scale_low=0.5,
                        )
                    except Exception as exc:
                        log.warning("insert_htmlbox indic p%d: %s", page_data.page_num, exc)
                        try:
                            page.insert_textbox(
                                block.rect, translated,
                                fontfile=unicode_font, fontname=unicode_alias,
                                fontsize=font_size, color=color,
                                align=fitz.TEXT_ALIGN_LEFT,
                            )
                        except Exception as e2:
                            log.warning("insert_textbox fallback p%d: %s", page_data.page_num, e2)
                else:
                    try:
                        page.insert_textbox(
                            block.rect, translated,
                            fontfile=unicode_font, fontname=unicode_alias,
                            fontsize=font_size, color=color,
                            align=fitz.TEXT_ALIGN_LEFT,
                        )
                    except Exception as exc:
                        log.warning("insert_textbox p%d: %s", page_data.page_num, exc)

            else:
                # Latin script: insert_htmlbox with standard CSS (no archive needed)
                latin_font = _resolve_font(font_name, font_flags)
                if latin_font.startswith("cour"):
                    css_family = "monospace"
                elif latin_font.startswith("ti"):
                    css_family = "serif"
                else:
                    css_family = "sans-serif"
                weight = "bold"   if is_bold   else "normal"
                style  = "italic" if is_italic else "normal"
                css = (
                    f"p{{font-family:{css_family};font-size:{font_size}pt;"
                    f"font-weight:{weight};font-style:{style};"
                    f"color:rgb({r_i},{g_i},{b_i});"
                    f"margin:0;padding:0;}}"
                )
                try:
                    page.insert_htmlbox(
                        block.rect, f"<p>{translated}</p>",
                        css=css, scale_low=0.5,
                    )
                except Exception as exc:
                    log.warning("insert_htmlbox latin p%d: %s", page_data.page_num, exc)

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
