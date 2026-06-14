"""
app/core/providers.py
=====================
Multi-LLM provider abstraction layer.
All providers implement the same translate_batch() interface.

Supported:
  claude   → Anthropic Claude
  openai   → OpenAI GPT
  gemini   → Google Gemini
  grok     → xAI Grok
  groq     → Groq Cloud (fast OSS inference)
  mistral  → Mistral AI
  cohere   → Cohere Command
  ollama   → Local Ollama
"""
from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Optional

log = logging.getLogger(__name__)

RETRY_ATTEMPTS = 3
RETRY_DELAY    = 2.0

# ── Shared utilities ──────────────────────────────────────────────────────────

def build_translation_prompt(texts: list[str], target_lang: str, source_lang: str) -> str:
    numbered = {str(i): t for i, t in enumerate(texts)}
    return (
        f"You are a professional document translator specialising in financial, legal, "
        f"and business documents.\n\n"
        f"Translate ALL of the following strings from {source_lang} to {target_lang}.\n\n"
        f"CRITICAL — these strings are individual text spans extracted from a PDF. "
        f"Each translation MUST be as SHORT as possible so it fits inside the same "
        f"fixed-width bounding box as the original. Use the most concise idiomatic "
        f"phrasing in {target_lang}; NEVER add explanations, context, or extra words.\n\n"
        f"STRICT RULES:\n"
        f"1. Return ONLY a valid JSON object — no markdown fences, no commentary.\n"
        f"2. Preserve the exact same numeric keys (0, 1, 2 ...).\n"
        f"3. Keep UNCHANGED in Latin script — do NOT transliterate ANY of these "
        f"into the target script under any circumstances:\n"
        f"   • Numbers, currencies (₹ £ $ €), percentages, dates\n"
        f"   • Period/quarter codes: FY25, FY26E, FY27E, H1FY26, Q1FY26, Q2FY25, "
        f"and any pattern matching Q#, FY##, Q#FY##, H#FY##\n"
        f"   • Growth ratios: QoQ, YoY, MoM\n"
        f"   • Financial metrics & acronyms: EBITDA, EBIT, EPS, CMP, NM, CAGR, bps, "
        f"P/E, P/B, RoNW, RoCE, RoIC, EV, ASP, PAT, PBT, ROCE, RONW, BEV, SoTP, "
        f"JLR, TML, CV, PV, FII, DII, OEM\n"
        f"   • Chart/table labels: LHS, RHS, H/L\n"
        f"   • Ticker symbols (e.g. TATMOT, RELIANCE, INFY) and CFA, MBA credentials\n"
        f"   • Email addresses and URLs\n"
        f"3b. For genuine brand/product NAMES (proper nouns used phonetically, "
        f"NOT financial abbreviations or metrics): if {target_lang} uses a non-Latin "
        f"script, write them PHONETICALLY in {target_lang} script. "
        f"Examples: 'TGS'→'टीजीएस', 'WhatsApp'→'व्हाट्सऐप', 'Screener'→'स्क्रीनर', "
        f"'Riise'→'राइस', 'Trendicator'→'ट्रेंडिकेटर'. "
        f"CRITICAL: Rule 3b NEVER applies to any financial metric, period code, "
        f"or abbreviation listed in rule 3 — those always stay in Latin.\n"
        f"3c. For English financial/trading/tech terms widely used as loanwords in "
        f"Indian languages, transliterate them phonetically — do NOT use a pure "
        f"native-language word. Examples for Hindi:\n"
        f"   trade→ट्रेड, trading→ट्रेडिंग, trader→ट्रेडर, "
        f"market→मार्केट, stock→स्टॉक, chart→चार्ट, signal→सिग्नल, "
        f"portfolio→पोर्टफोलियो, broker→ब्रोकर, analyst→एनालिस्ट, "
        f"target→टार्गेट, rating→रेटिंग, sector→सेक्टर, trend→ट्रेंड, "
        f"intraday→इंट्राडे, positional→पोज़िशनल, swing→स्विंग, "
        f"overnight→ओवरनाइट, exit→एग्जिट, entry→एंट्री, "
        f"alert→अलर्ट, score→स्कोर, hub→हब, live→लाइव, "
        f"webinar→वेबिनार, platform→प्लेटफॉर्म, module→मॉड्यूल, "
        f"digital→डिजिटल, auto→ऑटो, AI→एआई, SMS→एसएमएस, "
        f"email→ईमेल, capex→कैपेक्स, margin→मार्जिन.\n"
        f"4. Preserve capitalization style (ALL CAPS → ALL CAPS, where script allows).\n"
        f"5. Preserve leading/trailing spaces and punctuation exactly.\n"
        f"6. If a string is already in {target_lang}, return it unchanged.\n"
        f"7. Never merge, split, or reorder strings — one key in, one key out.\n"
        f"8. Target length: aim for at most the same character count as the source "
        f"string (or shorter). Never produce a translation longer than 1.5× the "
        f"source string length.\n\n"
        f"Input:\n{json.dumps(numbered, ensure_ascii=False)}\n\n"
        f"Output (JSON only):"
    )


_ZWS = chr(0x200B)  # Zero-width space — LLMs sometimes inject this between Indic syllables


def _clean_translation(text: str) -> str:
    """Remove artifacts that LLMs sometimes inject into Indic/non-Latin output."""
    text = text.replace(_ZWS, '')
    text = re.sub(r'[ \t]{2,}', ' ', text)    # Collapse multiple spaces/tabs
    return text.strip()


def parse_llm_response(raw: str, original_texts: list[str]) -> list[str]:
    """Parse JSON from LLM response. Multiple fallback strategies."""
    raw = raw.strip()

    # Strip markdown code fences
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()

    # Try direct parse
    try:
        parsed = json.loads(raw)
        return [_clean_translation(str(parsed.get(str(i), original_texts[i])))
                for i in range(len(original_texts))]
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from surrounding text
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            return [_clean_translation(str(parsed.get(str(i), original_texts[i])))
                    for i in range(len(original_texts))]
        except json.JSONDecodeError:
            pass

    log.warning("JSON parse failed — returning originals for this batch")
    return list(original_texts)


def with_retry(func):
    """Exponential-backoff retry decorator."""
    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt == RETRY_ATTEMPTS:
                    break
                wait = RETRY_DELAY * (2 ** (attempt - 1))
                log.warning(f"[{func.__qualname__}] attempt {attempt} failed: {exc}. Retrying in {wait:.1f}s")
                time.sleep(wait)
        log.error(f"[{func.__qualname__}] all {RETRY_ATTEMPTS} attempts failed: {last_exc}")
        raise last_exc
    wrapper.__name__ = func.__name__
    return wrapper


# ── Base class ────────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    name:          str = "base"
    default_model: str = ""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.model   = model or self.default_model
        self._init_client(**kwargs)

    def _init_client(self, **kwargs):
        """Override in subclasses to initialise the SDK client."""

    @abstractmethod
    def translate_batch(self, texts: list[str], target_lang: str, source_lang: str) -> list[str]:
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model!r})"


# ── Anthropic Claude ──────────────────────────────────────────────────────────

class ClaudeProvider(LLMProvider):
    name          = "claude"
    default_model = "claude-opus-4-5"

    def _init_client(self, **kwargs):
        import anthropic
        self.client = anthropic.Anthropic(api_key=self.api_key)

    @with_retry
    def translate_batch(self, texts, target_lang, source_lang):
        msg = self.client.messages.create(
            model      = self.model,
            max_tokens = 4096,
            messages   = [{"role": "user", "content": build_translation_prompt(texts, target_lang, source_lang)}],
        )
        return parse_llm_response(msg.content[0].text, texts)


# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    name          = "openai"
    default_model = "gpt-4o"

    def _init_client(self, **kwargs):
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)

    @with_retry
    def translate_batch(self, texts, target_lang, source_lang):
        resp = self.client.chat.completions.create(
            model       = self.model,
            max_tokens  = 4096,
            temperature = 0.1,
            messages    = [
                {"role": "system", "content": "You are a professional translator. Respond with valid JSON only."},
                {"role": "user",   "content": build_translation_prompt(texts, target_lang, source_lang)},
            ],
        )
        return parse_llm_response(resp.choices[0].message.content, texts)


# ── Google Gemini ─────────────────────────────────────────────────────────────

class GeminiProvider(LLMProvider):
    name          = "gemini"
    default_model = "gemini-2.0-flash"

    def _init_client(self, **kwargs):
        from google import genai
        from google.genai import types as gtypes
        self._gtypes = gtypes
        self.client  = genai.Client(api_key=self.api_key)

    @with_retry
    def translate_batch(self, texts, target_lang, source_lang):
        resp = self.client.models.generate_content(
            model    = self.model,
            contents = build_translation_prompt(texts, target_lang, source_lang),
            config   = self._gtypes.GenerateContentConfig(
                temperature        = 0.1,
                max_output_tokens  = 4096,
                response_mime_type = "application/json",
            ),
        )
        return parse_llm_response(resp.text, texts)


# ── xAI Grok ──────────────────────────────────────────────────────────────────

class GrokProvider(LLMProvider):
    name          = "grok"
    default_model = "grok-3"

    def _init_client(self, **kwargs):
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.x.ai/v1")

    @with_retry
    def translate_batch(self, texts, target_lang, source_lang):
        resp = self.client.chat.completions.create(
            model           = self.model,
            max_tokens      = 4096,
            temperature     = 0.1,
            response_format = {"type": "json_object"},
            messages        = [
                {"role": "system", "content": "You are a professional translator. Respond with valid JSON only."},
                {"role": "user",   "content": build_translation_prompt(texts, target_lang, source_lang)},
            ],
        )
        return parse_llm_response(resp.choices[0].message.content, texts)


# ── Groq Cloud ────────────────────────────────────────────────────────────────

class GroqProvider(LLMProvider):
    name          = "groq"
    default_model = "llama-3.3-70b-versatile"

    def _init_client(self, **kwargs):
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.groq.com/openai/v1")

    @with_retry
    def translate_batch(self, texts, target_lang, source_lang):
        resp = self.client.chat.completions.create(
            model           = self.model,
            max_tokens      = 4096,
            temperature     = 0.1,
            response_format = {"type": "json_object"},
            messages        = [
                {"role": "system", "content": "You are a professional translator. Respond with valid JSON only."},
                {"role": "user",   "content": build_translation_prompt(texts, target_lang, source_lang)},
            ],
        )
        return parse_llm_response(resp.choices[0].message.content, texts)


# ── Mistral ───────────────────────────────────────────────────────────────────

class MistralProvider(LLMProvider):
    name          = "mistral"
    default_model = "mistral-large-latest"

    def _init_client(self, **kwargs):
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.mistral.ai/v1")

    @with_retry
    def translate_batch(self, texts, target_lang, source_lang):
        resp = self.client.chat.completions.create(
            model           = self.model,
            max_tokens      = 4096,
            temperature     = 0.1,
            response_format = {"type": "json_object"},
            messages        = [
                {"role": "system", "content": "You are a professional translator. Respond with valid JSON only."},
                {"role": "user",   "content": build_translation_prompt(texts, target_lang, source_lang)},
            ],
        )
        return parse_llm_response(resp.choices[0].message.content, texts)


# ── Cohere ────────────────────────────────────────────────────────────────────

class CohereProvider(LLMProvider):
    name          = "cohere"
    default_model = "command-r-plus"

    def _init_client(self, **kwargs):
        import cohere
        self.client = cohere.Client(api_key=self.api_key)

    @with_retry
    def translate_batch(self, texts, target_lang, source_lang):
        resp = self.client.chat(
            model    = self.model,
            message  = build_translation_prompt(texts, target_lang, source_lang),
            preamble = "You are a professional translator. Respond with valid JSON only.",
            temperature = 0.1,
        )
        return parse_llm_response(resp.text, texts)


# ── Ollama (local) ────────────────────────────────────────────────────────────

class OllamaProvider(LLMProvider):
    name          = "ollama"
    default_model = "llama3"

    def _init_client(self, base_url: str = "http://localhost:11434", **kwargs):
        from openai import OpenAI
        self.client = OpenAI(api_key="ollama", base_url=f"{base_url.rstrip('/')}/v1")

    @with_retry
    def translate_batch(self, texts, target_lang, source_lang):
        resp = self.client.chat.completions.create(
            model           = self.model,
            max_tokens      = 4096,
            temperature     = 0.1,
            response_format = {"type": "json_object"},
            messages        = [
                {"role": "system", "content": "You are a professional translator. Respond with valid JSON only."},
                {"role": "user",   "content": build_translation_prompt(texts, target_lang, source_lang)},
            ],
        )
        return parse_llm_response(resp.choices[0].message.content, texts)


# ── Registry ──────────────────────────────────────────────────────────────────

PROVIDERS: dict[str, type[LLMProvider]] = {
    "claude":  ClaudeProvider,
    "openai":  OpenAIProvider,
    "gemini":  GeminiProvider,
    "grok":    GrokProvider,
    "groq":    GroqProvider,
    "mistral": MistralProvider,
    "cohere":  CohereProvider,
    "ollama":  OllamaProvider,
}

PROVIDER_MODELS: dict[str, list[str]] = {
    "claude":  ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5-20251001"],
    "openai":  ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    "gemini":  ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
    "grok":    ["grok-3", "grok-3-mini", "grok-2"],
    "groq":    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    "mistral": ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"],
    "cohere":  ["command-r-plus", "command-r", "command"],
    "ollama":  ["llama3", "llama3.1", "mistral", "gemma2", "phi3", "qwen2"],
}

ENV_KEYS: dict[str, str] = {
    "claude":  "ANTHROPIC_API_KEY",
    "openai":  "OPENAI_API_KEY",
    "gemini":  "GEMINI_API_KEY",
    "grok":    "GROK_API_KEY",
    "groq":    "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere":  "COHERE_API_KEY",
    "ollama":  "",
}


def get_provider(
    name:    str,
    api_key: Optional[str] = None,
    model:   Optional[str] = None,
    config=None,
    **kwargs,
) -> LLMProvider:
    """
    Factory. Resolves API key priority:
      1. Explicit api_key argument
      2. Flask app config (if provided)
      3. Environment variable
    """
    import os
    name = name.lower().strip()
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider '{name}'. Available: {', '.join(PROVIDERS)}")

    if api_key is None and name != "ollama":
        env_var = ENV_KEYS.get(name, "")
        if env_var:
            # Try Flask config first, then os.environ
            if config:
                api_key = getattr(config, env_var, None) or os.environ.get(env_var)
            else:
                api_key = os.environ.get(env_var)

    if name == "ollama" and "base_url" not in kwargs:
        if config:
            kwargs["base_url"] = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")
        else:
            kwargs["base_url"] = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    return PROVIDERS[name](api_key=api_key, model=model, **kwargs)
