"""
LatAm Localization Service.

Provides:
  - Language detection
  - Pharma-domain translation (English ↔ Spanish/Portuguese)
  - Locale-aware text formatting
"""

import re
import time
from typing import Optional

from vinci_core.engine import engine


# ── Supported locales ─────────────────────────────────────────────────────────
SUPPORTED_LOCALES = {
    "es-MX": "Spanish (Mexico)",
    "es-CO": "Spanish (Colombia)",
    "es-AR": "Spanish (Argentina)",
    "pt-BR": "Portuguese (Brazil)",
    "en-US": "English (US)",
}

# ── Simple keyword-based language detection ───────────────────────────────────
_PORTUGUESE_SIGNALS = [
    r"\bque\b", r"\bpara\b", r"\bcom\b", r"\bnão\b", r"\buma\b", r"\bem\b",
    r"\bdo\b", r"\bda\b", r"\bao\b", r"\bde\b", r"\bse\b", r"\bpor\b",
    r"\bsão\b", r"\bna\b", r"\bno\b", r"\bvocê\b", r"\bele\b", r"\bela\b",
    r"\bmedicamento\b", r"\bdosagem\b", r"\bpaciente\b", r"\bensaio\b",
]
_SPANISH_SIGNALS = [
    r"\bque\b", r"\bpara\b", r"\bcon\b", r"\bno\b", r"\buna\b", r"\ben\b",
    r"\bdel\b", r"\bde\b", r"\bal\b", r"\bse\b", r"\bpor\b", r"\bson\b",
    r"\bla\b", r"\bel\b", r"\blos\b", r"\blas\b", r"\busted\b", r"\bél\b",
    r"\bmedicamento\b", r"\bdosis\b", r"\bpaciente\b", r"\bensayo\b",
]
_PORTUGUESE_UNIQUE = [r"\bnão\b", r"\bvocê\b", r"\bsão\b", r"\buma\b", r"\bcom\b"]
_SPANISH_UNIQUE = [r"\bno\b", r"\busted\b", r"\bson\b", r"\bcon\b", r"\buna\b"]


def detect_language(text: str) -> str:
    """
    Heuristic language detection.

    Returns: "pt-BR", "es", or "en" (English default).
    """
    lower = text.lower()

    pt_score = sum(1 for p in _PORTUGUESE_SIGNALS if re.search(p, lower))
    pt_unique = sum(1 for p in _PORTUGUESE_UNIQUE if re.search(p, lower))
    es_score = sum(1 for p in _SPANISH_SIGNALS if re.search(p, lower))
    es_unique = sum(1 for p in _SPANISH_UNIQUE if re.search(p, lower))

    # Unique markers carry double weight
    pt_total = pt_score + pt_unique * 2
    es_total = es_score + es_unique * 2

    if pt_total > es_total and pt_total >= 3:
        return "pt-BR"
    if es_total >= 3:
        return "es"
    return "en"


# ── Translation prompts ───────────────────────────────────────────────────────
_TRANSLATION_SYSTEM = {
    "es-MX": (
        "You are an expert pharmaceutical regulatory translator specializing in "
        "Mexican Spanish (es-MX). Translate the following text to Mexican Spanish "
        "with pharmaceutical/regulatory domain accuracy. "
        "Use COFEPRIS regulatory terminology. "
        "Keep scientific names (INN/DCI), chemical formulas, and numeric values unchanged. "
        "Translate ONLY the provided text — no explanations."
    ),
    "es-CO": (
        "You are an expert pharmaceutical regulatory translator specializing in "
        "Colombian Spanish (es-CO). Translate the following text to Colombian Spanish "
        "with pharmaceutical/regulatory domain accuracy. "
        "Use INVIMA regulatory terminology. "
        "Keep scientific names (INN/DCI), chemical formulas, and numeric values unchanged. "
        "Translate ONLY the provided text — no explanations."
    ),
    "es-AR": (
        "You are an expert pharmaceutical regulatory translator specializing in "
        "Argentine Spanish (es-AR). Translate the following text to Argentine Spanish "
        "with pharmaceutical/regulatory domain accuracy. "
        "Use ANMAT regulatory terminology. "
        "Keep scientific names (DCI), chemical formulas, and numeric values unchanged. "
        "Translate ONLY the provided text — no explanations."
    ),
    "pt-BR": (
        "You are an expert pharmaceutical regulatory translator specializing in "
        "Brazilian Portuguese (pt-BR). Translate the following text to Brazilian Portuguese "
        "with pharmaceutical/regulatory domain accuracy. "
        "Use ANVISA regulatory terminology (DCB for drug names). "
        "Keep scientific names (DCB/DCI), chemical formulas, and numeric values unchanged. "
        "Translate ONLY the provided text — no explanations."
    ),
}


async def translate_text(
    text: str,
    target_locale: str,
    source_locale: Optional[str] = None,
) -> dict:
    """
    Translate text to the target locale with pharma domain accuracy.

    Args:
        text: Source text to translate
        target_locale: e.g. "es-MX", "pt-BR", "es-CO", "es-AR"
        source_locale: Source language hint (auto-detected if None)

    Returns:
        dict with translated_text, target_locale, source_locale,
              confidence, latency_ms, character_count
    """
    start_ms = time.monotonic()

    if target_locale not in _TRANSLATION_SYSTEM:
        # No translation needed for English or unsupported locale
        return {
            "translated_text": text,
            "target_locale": target_locale,
            "source_locale": source_locale or "en",
            "confidence": 1.0,
            "latency_ms": 0,
            "character_count": len(text),
            "note": f"Target locale '{target_locale}' not supported; text returned as-is",
        }

    detected_source = source_locale or detect_language(text)

    # Skip if same language family
    if target_locale.startswith("es") and detected_source in ("es", "es-MX", "es-CO", "es-AR"):
        return {
            "translated_text": text,
            "target_locale": target_locale,
            "source_locale": detected_source,
            "confidence": 0.95,
            "latency_ms": 0,
            "character_count": len(text),
            "note": "Source and target are in the same language family; minimal translation applied",
        }

    system_prompt = _TRANSLATION_SYSTEM[target_locale]
    prompt = f"{system_prompt}\n\nText to translate:\n{text}"

    response = await engine.run(
        prompt=prompt,
        layer="pharma",
        use_rag=False,
    )

    elapsed_ms = int((time.monotonic() - start_ms) * 1000)

    return {
        "translated_text": response.content,
        "target_locale": target_locale,
        "source_locale": detected_source,
        "confidence": 0.92,   # domain-specific translation confidence
        "latency_ms": elapsed_ms,
        "character_count": len(text),
        "model": response.model,
    }


async def batch_translate(
    text: str,
    target_locales: list,
) -> dict:
    """
    Translate a single text to multiple locales in sequence.

    Returns:
        dict mapping locale → translation result
    """
    results = {}
    for locale in target_locales:
        results[locale] = await translate_text(text, locale)
    return results
