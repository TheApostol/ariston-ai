"""
Provider Status Endpoint — GET /api/v1/providers/status

Tests each AI provider with a minimal request (max 5 tokens) and returns
real-time status. Results are cached for 60 seconds to avoid hammering APIs.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Dict, Any

from fastapi import APIRouter

logger = logging.getLogger("ariston.providers")

router = APIRouter(prefix="/providers", tags=["Providers"])

# ---------------------------------------------------------------------------
# In-memory 60-second cache
# ---------------------------------------------------------------------------

_cache: Dict[str, Any] = {}
_cache_ts: float = 0.0
_CACHE_TTL = 60.0


# ---------------------------------------------------------------------------
# Provider test helpers
# ---------------------------------------------------------------------------

async def _test_anthropic() -> Dict[str, Any]:
    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=5,
            messages=[{"role": "user", "content": "hi"}],
        )
        return {"status": "live", "model": "claude-sonnet-4-6"}
    except Exception as e:
        err = str(e).lower()
        if "quota" in err or "rate" in err or "overloaded" in err:
            return {"status": "quota_exhausted", "model": "claude-sonnet-4-6"}
        if "api_key" in err or "authentication" in err or "invalid" in err:
            return {"status": "invalid_key", "model": "claude-sonnet-4-6"}
        return {"status": "error", "model": "claude-sonnet-4-6", "detail": str(e)[:80]}


async def _test_gemini() -> Dict[str, Any]:
    try:
        import google.generativeai as genai
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"status": "no_key", "model": "gemini-2.0-flash"}
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        model.generate_content("hi", generation_config={"max_output_tokens": 5})
        return {"status": "live", "model": "gemini-2.0-flash"}
    except Exception as e:
        err = str(e).lower()
        if "quota" in err or "429" in err or "resource exhausted" in err:
            return {"status": "quota_exhausted", "model": "gemini-2.0-flash"}
        if "api_key" in err or "invalid" in err or "403" in err:
            return {"status": "invalid_key", "model": "gemini-2.0-flash"}
        return {"status": "error", "model": "gemini-2.0-flash", "detail": str(e)[:80]}


async def _test_openai() -> Dict[str, Any]:
    try:
        import openai as _openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {"status": "no_key"}
        client = _openai.AsyncOpenAI(api_key=api_key)
        await client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=5,
            messages=[{"role": "user", "content": "hi"}],
        )
        return {"status": "live", "model": "gpt-4o-mini"}
    except Exception as e:
        err = str(e).lower()
        if "401" in err or "invalid" in err or "incorrect" in err:
            return {"status": "invalid_key"}
        if "quota" in err or "429" in err:
            return {"status": "quota_exhausted"}
        return {"status": "error", "detail": str(e)[:80]}


async def _test_openrouter() -> Dict[str, Any]:
    try:
        import httpx
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return {"status": "no_key"}
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "openai/gpt-4o-mini",
                    "max_tokens": 5,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        if r.status_code == 401:
            return {"status": "invalid_key"}
        if r.status_code == 429:
            return {"status": "quota_exhausted"}
        r.raise_for_status()
        return {"status": "live", "model": "openai/gpt-4o-mini"}
    except Exception as e:
        err = str(e).lower()
        if "401" in err or "invalid" in err:
            return {"status": "invalid_key"}
        return {"status": "error", "detail": str(e)[:80]}


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get("/status")
async def provider_status() -> Dict[str, Any]:
    """
    Test each provider with a 5-token request and return real-time status.
    Results are cached for 60 seconds.

    Returns:
        {
          "anthropic":   {"status": "live",            "model": "claude-sonnet-4-6"},
          "gemini":      {"status": "quota_exhausted", "model": "gemini-2.0-flash"},
          "openai":      {"status": "invalid_key"},
          "openrouter":  {"status": "invalid_key"},
          "cached":      false,
          "checked_at":  "2026-04-12T12:00:00"
        }
    """
    global _cache, _cache_ts

    now = time.time()
    if _cache and (now - _cache_ts) < _CACHE_TTL:
        return {**_cache, "cached": True}

    # Run all checks in parallel with a 10-second overall timeout
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                _test_anthropic(),
                _test_gemini(),
                _test_openai(),
                _test_openrouter(),
                return_exceptions=True,
            ),
            timeout=12.0,
        )
    except asyncio.TimeoutError:
        results = [
            {"status": "timeout"},
            {"status": "timeout"},
            {"status": "timeout"},
            {"status": "timeout"},
        ]

    def _safe(r):
        if isinstance(r, Exception):
            return {"status": "error", "detail": str(r)[:80]}
        return r

    status = {
        "anthropic":  _safe(results[0]),
        "gemini":     _safe(results[1]),
        "openai":     _safe(results[2]),
        "openrouter": _safe(results[3]),
        "cached":     False,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
    }

    _cache    = status
    _cache_ts = now

    return status
