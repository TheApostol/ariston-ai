import asyncio
import logging
import time
from typing import Any, Dict

import httpx
from fastapi import APIRouter, HTTPException
from vinci_core.engine import engine
from vinci_core.schemas import CompletionRequest, AIResponse

logger = logging.getLogger("ariston.api.vinci")
router = APIRouter()

# Shallow probe timeout in seconds — kept short so health checks are fast
_PROBE_TIMEOUT = 5.0


async def _probe_anthropic() -> str:
    """Check Anthropic API reachability (unauthenticated HEAD to status page)."""
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
            resp = await client.get("https://status.anthropic.com/api/v2/status.json")
        data = resp.json()
        indicator = data.get("status", {}).get("indicator", "unknown")
        return "ok" if indicator in ("none", "minor") else "degraded"
    except Exception:
        return "unreachable"


async def _probe_openrouter() -> str:
    """Check OpenRouter API reachability via its public models endpoint."""
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
            resp = await client.get("https://openrouter.ai/api/v1/models")
        return "ok" if resp.status_code == 200 else "degraded"
    except Exception:
        return "unreachable"


async def _probe_ollama() -> str:
    """Check local Ollama reachability."""
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
            resp = await client.get("http://localhost:11434/api/tags")
        return "ok" if resp.status_code == 200 else "degraded"
    except Exception:
        return "unreachable"


@router.get(
    "/health",
    summary="Health check with provider probing",
    description=(
        "Probes each configured AI provider and returns its reachability status. "
        "Returns HTTP 200 even when providers are degraded — callers should inspect "
        "the response body. Returns HTTP 503 only when the engine itself is broken."
    ),
)
async def health_check() -> Dict[str, Any]:
    """
    Returns:
        {
            "status": "ok" | "degraded",
            "providers": {
                "anthropic":   "ok" | "degraded" | "unreachable",
                "openrouter":  "ok" | "degraded" | "unreachable",
                "ollama":      "ok" | "degraded" | "unreachable"
            },
            "latency_ms": <int>
        }
    """
    start = time.monotonic()
    anthropic_status, openrouter_status, ollama_status = await asyncio.gather(
        _probe_anthropic(),
        _probe_openrouter(),
        _probe_ollama(),
    )
    latency_ms = round((time.monotonic() - start) * 1000)

    providers = {
        "anthropic": anthropic_status,
        "openrouter": openrouter_status,
        "ollama": ollama_status,
    }
    overall = "ok" if all(v == "ok" for v in providers.values()) else "degraded"

    logger.info(
        '{"event":"health_check","status":"%s","latency_ms":%d}',
        overall, latency_ms,
    )
    return {"status": overall, "providers": providers, "latency_ms": latency_ms}



@router.post(
    "/vinci/complete",
    response_model=AIResponse,
    summary="Ariston AI completion endpoint",
    description=(
        "Runs a prompt through Ariston AI using the selected model and layer. "
        "Supports local (Ollama) and free cloud (OpenRouter) inference."
    ),
)
async def complete(request: CompletionRequest):
    """
    Request body:
    {
        "prompt": "...",
        "model": "openrouter/free" | "ollama",
        "layer": "base" | "pharma" | "clinical" | "data" | "radiology" | "general",
        "context": {}
    }
    """
    try:
        result = await engine.run(**request.model_dump())
        return result
    except Exception as e:
        logger.error(
            '{"event":"api_error","endpoint":"/vinci/complete","error_type":"%s"}',
            type(e).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected server error occurred. Please try again.",
        )
